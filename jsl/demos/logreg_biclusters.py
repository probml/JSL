# Bayesian logistic regression in 2d for 2 class problem
# We compare MCMC to Laplace

# Dependencies:
#     * !pip install git+https://github.com/blackjax-devs/blackjax.git


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from blackjax import rmh
from jax import random
from functools import partial
from jax.scipy.optimize import minimize
from sklearn.datasets import make_biclusters
from jax.scipy.stats import norm


def sigmoid(x): return jnp.exp(x) / (1 + jnp.exp(x))
def log_sigmoid(z): return z - jnp.log1p(jnp.exp(z))


def plot_posterior_predictive(ax, X, Xspace, Zspace, title, colors, cmap="RdBu_r"):
    ax.contourf(*Xspace, Zspace, cmap=cmap, alpha=0.7, levels=20)
    ax.scatter(*X.T, c=colors, edgecolors="gray", s=80)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()


def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def E_base(w, Phi, y, alpha):
    # Energy is the log joint
    an = Phi @ w
    log_an = log_sigmoid(an)
    log_likelihood_term = y * log_an + (1 - y) * jnp.log(1 - sigmoid(an))
    log_prior_term = -(alpha * w @ w / 2)
    return log_prior_term + log_likelihood_term.sum()


def mcmc_logistic_posterior_sample(key, Phi, y, alpha=1.0, init_noise=1.0,
                                   n_samples=5_000, burnin=300, sigma_mcmc=0.8):
    """
    Sample from the posterior distribution of the weights
    of a 2d binary logistic regression model p(y=1|x,w) = sigmoid(w'x),
    using the random walk Metropolis-Hastings algorithm. 
    """
    _, ndims = Phi.shape
    key, key_init = random.split(key)
    w0 = random.multivariate_normal(key, jnp.zeros(ndims), jnp.eye(ndims) * init_noise)
    energy = partial(E_base, Phi=Phi, y=y, alpha=alpha)
    mcmc_kernel = rmh(energy, sigma=jnp.ones(ndims) * sigma_mcmc)
    initial_state = mcmc_kernel.init(w0)

    states = inference_loop(key_init, mcmc_kernel.step, initial_state, n_samples)
    chains = states.position[burnin:, :]
    return chains

def laplace_posterior(key, Phi, y, alpha=1.0, init_noise=1.0):
    N, M = Phi.shape
    w0 = random.multivariate_normal(key, jnp.zeros(M), jnp.eye(M) * init_noise)
    E = lambda w: -E_base(w, Phi, y, alpha) / len(y)
    res = minimize(E, w0, method="BFGS")
    w_laplace = res.x
    SN = jax.hessian(E)(w_laplace)
    return w_laplace, SN


def main():
    ## Data generating process
    n_datapoints = 50
    m = 2
    X, rows, _ = make_biclusters((n_datapoints, m), 2,
                                    noise=0.6, random_state=3141,
                                    minval=-4, maxval=4)
    # whether datapoints belong to class 1
    y = rows[0] * 1.0

    Phi = jnp.c_[jnp.ones(n_datapoints)[:, None], X]
    N, M = Phi.shape

    colors = ["black" if el else "white" for el in y]

    # Predictive domain
    xmin, ymin = X.min(axis=0) - 0.1
    xmax, ymax = X.max(axis=0) + 0.1
    step = 0.1
    Xspace = jnp.mgrid[xmin:xmax:step, ymin:ymax:step]
    _, nx, ny = Xspace.shape
    Phispace = jnp.concatenate([jnp.ones((1, nx, ny)), Xspace])

    key = random.PRNGKey(314)

    ## Laplace
    alpha = 2.0
    w_laplace, SN = laplace_posterior(key, Phi, y, alpha=alpha)

    ### MCMC Approximation
    chains = mcmc_logistic_posterior_sample(key, Phi, y, alpha=alpha)
    Z_mcmc = sigmoid(jnp.einsum("mij,sm->sij", Phispace, chains))
    Z_mcmc = Z_mcmc.mean(axis=0)

    ### *** Ploting surface predictive distribution ***
    colors = ["black" if el else "white" for el in y]
    dict_figures = {}
    key = random.PRNGKey(31415)
    nsamples = 5000

    # Laplace surface predictive distribution
    laplace_samples = random.multivariate_normal(key, w_laplace, SN, (nsamples,))
    Z_laplace = sigmoid(jnp.einsum("mij,sm->sij", Phispace, laplace_samples))
    Z_laplace = Z_laplace.mean(axis=0)

    fig_laplace, ax = plt.subplots()
    title = "Laplace Predictive distribution"
    plot_posterior_predictive(ax, X, Xspace, Z_laplace, title, colors)
    dict_figures["logistic_regression_surface_laplace"] = fig_laplace

    # MCMC surface predictive distribution
    fig_mcmc, ax = plt.subplots()
    title = "MCMC Predictive distribution"
    plot_posterior_predictive(ax, X, Xspace, Z_mcmc, title, colors)
    dict_figures["logistic_regression_surface_mcmc"] = fig_mcmc


    # *** Plotting posterior marginals of weights ***
    for i in range(M):
        fig_weights_marginals, ax = plt.subplots()
        mean_laplace, std_laplace = w_laplace[i], jnp.sqrt(SN[i, i])
        mean_mcmc, std_mcmc = chains[:, i].mean(), chains[:, i].std()

        x = jnp.linspace(mean_laplace - 4 * std_laplace, mean_laplace + 4 * std_laplace, 500)
        ax.plot(x, norm.pdf(x, mean_laplace, std_laplace), label="posterior (Laplace)", linestyle="dotted")
        ax.plot(x, norm.pdf(x, mean_mcmc, std_mcmc), label="posterior (MCMC)", linestyle="dashed")
        ax.legend()
        ax.set_title(f"Posterior marginals of weights ({i})")
        dict_figures[f"logistic_regression_weights_marginals_w{i}"] = fig_weights_marginals


    print("MCMC weights")
    w_mcmc = chains.mean(axis=0)
    print(w_mcmc, end="\n"*2)

    print("Laplace weights")
    print(w_laplace, end="\n"*2)

    dict_data = {
        "X": X,
        "y": y,
        "Xspace": Xspace,
        "Phi": Phi,
        "Phispace": Phispace,
        "w_laplace": w_laplace,
        "cov_laplace": SN
    }

    return dict_figures, dict_data


if __name__ == "__main__":
    from jsl.demos.plot_utils import savefig
    figs, data = main()
    savefig(figs)
    plt.show()
