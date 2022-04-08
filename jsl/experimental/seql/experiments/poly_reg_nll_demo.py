import jax.numpy as jnp
from jax import random

import optax
from jaxopt import ScipyMinimize

from functools import partial
from matplotlib import pyplot as plt

from jsl.experimental.seql.agents.bayesian_lin_reg_agent import bayesian_reg
from jsl.experimental.seql.agents.bfgs_agent import bfgs_agent
from jsl.experimental.seql.agents.blackjax_nuts_agent import blackjax_nuts_agent
from jsl.experimental.seql.agents.kf_agent import kalman_filter_reg
from jsl.experimental.seql.agents.laplace_agent import laplace_agent
from jsl.experimental.seql.agents.lbfgs_agent import lbfgs_agent
from jsl.experimental.seql.agents.sgd_agent import sgd_agent
from jsl.experimental.seql.agents.sgmcmc_sgld_agent import sgld_agent
from jsl.experimental.seql.environments.base import make_evenly_spaced_x_sampler, \
    make_random_poly_regression_environment
from jsl.experimental.seql.experiments.experiment_utils import run_experiment
from jsl.experimental.seql.experiments.plotting import colors
from jsl.experimental.seql.utils import mse, train

plt.style.use("seaborn-poster")


def model_fn(w, x):
    return x @ w


def logprior_fn(params):
    return 0.


def negative_mean_square_error(params, inputs, outputs, model_fn, strength=0.):
    return -penalized_objective_fn(params, inputs, outputs, model_fn, strength=0.)


def penalized_objective_fn(params, inputs, outputs, model_fn, strength=0.):
    return mse(params, inputs, outputs, model_fn) + strength * jnp.sum(params ** 2)


def energy_fn(params, data, model_fn, strength=0.):
    return mse(params, *data, model_fn) + strength * jnp.sum(params ** 2)


losses = []


def callback_fn(agent, env, agent_name, **kwargs):
    global losses
    if kwargs["t"] == 0:
        losses = []
    belief = kwargs["belief_state"]
    nfeatures = kwargs["nfeatures"]
    out = 1

    inputs = env.X_test.reshape((-1, nfeatures))
    outputs = env.y_test.reshape((-1, out))
    predictions, _ = agent.predict(belief, inputs)
    loss = jnp.mean(jnp.power(predictions - outputs, 2))
    losses.append(loss)
    if kwargs["t"] == 9:
        ax = kwargs["ax"]
        ax.plot(losses, color=colors[agent_name])
        ax.set_title(agent_name.upper())
        plt.tight_layout()
        plt.savefig("asas.png")


def initialize_params(agent_name, **kwargs):
    nfeatures = kwargs["degree"] + 1
    mu0 = jnp.zeros((nfeatures, 1))
    if agent_name in ["exact bayes", "kf"]:
        mu0 = jnp.zeros((nfeatures, 1))
        Sigma0 = jnp.eye(nfeatures)
        initial_params = (mu0, Sigma0)
    else:
        initial_params = (mu0,)

    return initial_params


def main():
    key = random.PRNGKey(0)

    min_val, max_val = -3, 3
    x_test_generator = make_evenly_spaced_x_sampler(max_val,
                                                    use_bias=False,
                                                    min_val=min_val)

    degree = 3
    ntrain = 50
    ntest = 50
    batch_size = 5
    obs_noise = 1.

    env_key, nuts_key, sgld_key = random.split(key, 3)
    env = lambda batch_size: make_random_poly_regression_environment(env_key,
                                                                     degree,
                                                                     ntrain,
                                                                     ntest,
                                                                     obs_noise=obs_noise,
                                                                     train_batch_size=batch_size,
                                                                     test_batch_size=batch_size,
                                                                     x_test_generator=x_test_generator)

    nsteps = 10

    buffer_size = ntrain

    kf = kalman_filter_reg(obs_noise)

    bayes = bayesian_reg(buffer_size, obs_noise)
    batch_bayes = bayesian_reg(ntrain, obs_noise)

    optimizer = optax.adam(1e-1)

    nepochs = 4
    sgd = sgd_agent(mse,
                    model_fn,
                    optimizer=optimizer,
                    obs_noise=obs_noise,
                    nepochs=nepochs,
                    buffer_size=buffer_size)

    batch_sgd = sgd_agent(mse,
                          model_fn,
                          optimizer=optimizer,
                          obs_noise=obs_noise,
                          buffer_size=buffer_size,
                          nepochs=nepochs * nsteps)

    nsamples, nwarmup = 200, 100
    nuts = blackjax_nuts_agent(nuts_key,
                               negative_mean_square_error,
                               model_fn,
                               nsamples=nsamples,
                               nwarmup=nwarmup,
                               obs_noise=obs_noise,
                               buffer_size=buffer_size)

    batch_nuts = blackjax_nuts_agent(nuts_key,
                                     negative_mean_square_error,
                                     model_fn,
                                     nsamples=nsamples * nsteps,
                                     nwarmup=nwarmup,
                                     obs_noise=obs_noise,
                                     buffer_size=buffer_size)

    partial_logprob_fn = partial(negative_mean_square_error,
                                 model_fn=model_fn)
    dt = 1e-4
    sgld = sgld_agent(sgld_key,
                      partial_logprob_fn,
                      logprior_fn,
                      model_fn,
                      dt=dt,
                      batch_size=batch_size,
                      nsamples=nsamples,
                      obs_noise=obs_noise,
                      buffer_size=buffer_size)

    dt = 1e-5
    batch_sgld = sgld_agent(sgld_key,
                            partial_logprob_fn,
                            logprior_fn,
                            model_fn,
                            dt=dt,
                            batch_size=batch_size,
                            nsamples=nsamples * nsteps,
                            obs_noise=obs_noise,
                            buffer_size=buffer_size)

    tau = 1.
    strength = obs_noise / tau
    partial_objective_fn = partial(penalized_objective_fn, strength=strength)

    bfgs = bfgs_agent(partial_objective_fn,
                      obs_noise=obs_noise,
                      buffer_size=buffer_size)

    batch_bfgs = bfgs_agent(partial_objective_fn,
                            obs_noise=obs_noise,
                            buffer_size=buffer_size)

    lbfgs = lbfgs_agent(partial_objective_fn,
                        obs_noise=obs_noise,
                        history_size=buffer_size)

    batch_lbfgs = lbfgs_agent(partial_objective_fn,
                              obs_noise=obs_noise,
                              history_size=buffer_size)

    energy_fn = partial(partial_objective_fn, model_fn=model_fn)
    solver = ScipyMinimize(fun=energy_fn, method="BFGS")
    laplace = laplace_agent(solver,
                            energy_fn,
                            model_fn,
                            obs_noise=obs_noise,
                            buffer_size=buffer_size)

    agents = {
        "kf": kf,
        "exact bayes": bayes,
        "sgd": sgd,
        "laplace": laplace,
        "bfgs": bfgs,
        "lbfgs": lbfgs,
        "nuts": nuts,
        "sgld": sgld,
    }

    batch_agents = {
        "kf": kf,
        "exact bayes": batch_bayes,
        "sgd": batch_sgd,
        "laplace": laplace,
        "bfgs": batch_bfgs,
        "lbfgs": batch_lbfgs,
        "nuts": batch_nuts,
        "sgld": batch_sgld,
    }
    nrows = len(agents)
    ncols = 1
    timesteps = list(range(nsteps))

    run_experiment(agents,
                   env,
                   initialize_params,
                   batch_size,
                   ntrain,
                   nsteps,
                   nrows,
                   ncols,
                   callback_fn=callback_fn,
                   degree=degree,
                   obs_noise=obs_noise,
                   timesteps=timesteps,
                   key=key,
                   nfeatures=degree + 1,
                   batch_agents=batch_agents)


if __name__ == "__main__":
    main()
