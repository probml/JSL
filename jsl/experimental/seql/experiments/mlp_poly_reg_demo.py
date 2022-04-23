import jax.numpy as jnp
from jax import random, tree_leaves, tree_map

import optax
import flax.linen as nn
from jaxopt import ScipyMinimize
from functools import partial
from matplotlib import pyplot as plt

from jsl.experimental.seql.agents.bfgs_agent import bfgs_agent
from jsl.experimental.seql.agents.blackjax_nuts_agent import blackjax_nuts_agent
from jsl.experimental.seql.agents.laplace_agent import laplace_agent
from jsl.experimental.seql.agents.lbfgs_agent import lbfgs_agent
from jsl.experimental.seql.agents.sgd_agent import sgd_agent
from jsl.experimental.seql.agents.sgmcmc_sgld_agent import sgld_agent
from jsl.experimental.seql.environments.base import make_evenly_spaced_x_sampler, \
    make_random_poly_regression_environment
from jsl.experimental.seql.experiments.experiment_utils import run_experiment
from jsl.experimental.seql.experiments.plotting import plot_regression_posterior_predictive
from jsl.experimental.seql.utils import mean_squared_error


class MLP(nn.Module):
    nclasses: int

    @nn.compact
    def __call__(self, x
                 ):
        x = nn.tanh(nn.Dense(50)(x))
        x = nn.Dense(self.nclasses)(x)
        return x


model = MLP(1)
model_fn = model.apply


def logprior_fn(params):
    leaves = tree_leaves(params)
    return -sum(tree_map(lambda x: jnp.sum(x ** 2), leaves))


def loglikelihood_fn(params, x, y, model_fn):
    return -mean_squared_error(params, x, y, model_fn) * len(x)


def logjoint_fn(params, x, y, model_fn, strength=0.01):
    return loglikelihood_fn(params, x, y, model_fn) + strength * logprior_fn(params)


def neg_logjoint_fn(params, inputs, outputs, model_fn, strength=0.1):
    return -logjoint_fn(params, inputs, outputs, model_fn, strength=strength)


def callback_fn(agent, env, agent_name, **kwargs):
    if "subplot_idx" not in kwargs:
        subplot_idx = kwargs["t"] + kwargs["idx"] * kwargs["ncols"] + 1
    else:
        subplot_idx = kwargs["subplot_idx"]

    ax = kwargs["fig"].add_subplot(kwargs["nrows"],
                                   kwargs["ncols"],
                                   subplot_idx)

    belief = kwargs["belief_state"]

    plot_regression_posterior_predictive(ax,
                                         agent,
                                         env,
                                         belief,
                                         t=kwargs["t"])
    if "title" in kwargs:
        ax.set_title(kwargs["title"], fontsize=32)
    else:
        ax.set_title("t={}".format(kwargs["t"]), fontsize=32)

    plt.tight_layout()
    plt.savefig("jaks.png")


def initialize_params(agent_name, **kwargs):
    batch = jnp.ones((1, kwargs["nfeatures"]))
    variables = kwargs["model"].init(kwargs["key"], batch)
    return variables


def main():
    key = random.PRNGKey(0)

    min_val, max_val = -3, 3
    x_test_generator = make_evenly_spaced_x_sampler(max_val,
                                                    use_bias=False,
                                                    min_val=min_val)

    degree = 3
    ntrain = 40
    ntest = 64
    train_batch_size = 4
    obs_noise = 1.

    env_key, nuts_key, sgld_key, model_key = random.split(key, 4)
    env = lambda batch_size: make_random_poly_regression_environment(env_key,
                                                                     degree,
                                                                     ntrain,
                                                                     ntest,
                                                                     obs_noise=obs_noise,
                                                                     train_batch_size=batch_size,
                                                                     x_test_generator=x_test_generator)

    nsteps = 10

    buffer_size = ntrain

    optimizer = optax.adam(1e-1)

    nepochs = 4
    sgd = sgd_agent(mean_squared_error,
                    model_fn,
                    optimizer=optimizer,
                    obs_noise=obs_noise,
                    nepochs=nepochs,
                    buffer_size=buffer_size)

    batch_sgd = sgd_agent(mean_squared_error,
                          model_fn,
                          optimizer=optimizer,
                          obs_noise=obs_noise,
                          buffer_size=ntrain,
                          nepochs=nepochs * nsteps)

    nsamples, nwarmup = 200, 100
    nuts = blackjax_nuts_agent(nuts_key,
                               logjoint_fn,
                               model_fn,
                               nsamples=nsamples,
                               nwarmup=nwarmup,
                               obs_noise=obs_noise,
                               nlast=0,
                               buffer_size=buffer_size)

    batch_nuts = blackjax_nuts_agent(nuts_key,
                                     logjoint_fn,
                                     model_fn,
                                     nsamples=nsamples * nsteps,
                                     nwarmup=nwarmup,
                                     obs_noise=obs_noise,
                                     buffer_size=ntrain)

    partial_logprob_fn = partial(logjoint_fn,
                                 model_fn=model_fn)
    dt = 1e-4
    sgld = sgld_agent(sgld_key,
                      partial_logprob_fn,
                      logprior_fn,
                      model_fn,
                      dt=dt,
                      batch_size=train_batch_size,
                      nsamples=nsamples,
                      obs_noise=obs_noise,
                      buffer_size=buffer_size)

    dt = 1e-5
    batch_sgld = sgld_agent(sgld_key,
                            partial_logprob_fn,
                            logprior_fn,
                            model_fn,
                            dt=dt,
                            batch_size=train_batch_size,
                            nsamples=nsamples * nsteps,
                            obs_noise=obs_noise,
                            buffer_size=ntrain)

    tau = 1.
    strength = obs_noise / tau
    partial_objective_fn = partial(neg_logjoint_fn, strength=strength)

    bfgs = bfgs_agent(partial_objective_fn,
                      model_fn=model_fn,
                      obs_noise=obs_noise,
                      buffer_size=buffer_size)

    batch_bfgs = bfgs_agent(partial_objective_fn,
                            model_fn=model_fn,
                            obs_noise=obs_noise,
                            buffer_size=buffer_size)

    lbfgs = lbfgs_agent(partial_objective_fn,
                        model_fn=model_fn,
                        obs_noise=obs_noise,
                        history_size=buffer_size)

    batch_lbfgs = lbfgs_agent(partial_objective_fn,
                              model_fn=model_fn,
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
        "sgd": sgd,
        "laplace": laplace,
        "bfgs": bfgs,
        "lbfgs": lbfgs,
        "nuts": nuts,
        "sgld": sgld,
    }

    batch_agents = {
        "sgd": batch_sgd,
        "laplace": laplace,
        "nuts": batch_nuts,
        "sgld": batch_sgld,
        "bfgs": batch_bfgs,
        "lbfgs": batch_lbfgs
    }

    timesteps = list(range(nsteps))
    nrows = len(agents)
    ncols = len(timesteps) + 1

    run_experiment(agents,
                   env,
                   initialize_params,
                   train_batch_size,
                   ntrain,
                   nsteps,
                   nrows,
                   ncols,
                   callback_fn,
                   nfeatures=4,
                   model=model,
                   key=model_key,
                   obs_noise=obs_noise,
                   batch_agents=batch_agents)


if __name__ == "__main__":
    main()
