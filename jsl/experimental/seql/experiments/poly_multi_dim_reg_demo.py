import jax.numpy as jnp
from jax import random

import optax
from jaxopt import ScipyMinimize

from functools import partial
from matplotlib import pyplot as plt

from jsl.experimental.seql.experiments.experiment_utils import run_experiment
from jsl.experimental.seql.experiments.plotting import colors

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
from jsl.experimental.seql.utils import mean_squared_error

plt.style.use("seaborn-poster")


def model_fn(w, x):
    return x @ w


def logprior_fn(params):
    return 0.


def negative_mean_square_error(params, inputs, outputs, model_fn, strength=0.):
    return -penalized_objective_fn(params, inputs, outputs, model_fn, strength=0.)


def penalized_objective_fn(params, inputs, outputs, model_fn, strength=0.):
    return mean_squared_error(params, inputs, outputs, model_fn) + strength * jnp.sum(params ** 2)


def energy_fn(params, data, model_fn, strength=0.):
    return mean_squared_error(params, *data, model_fn) + strength * jnp.sum(params ** 2)


losses = []


def callback_fn(agent, env, agent_name, **kwargs):
    global loss

    belief = kwargs["belief_state"]
    nfeatures = kwargs["degree"] + 1
    out = kwargs["nout"]

    inputs = env.X_test.reshape((-1, nfeatures))
    outputs = env.y_test.reshape((-1, out))
    predictions, _ = agent.predict(belief, inputs)
    loss = jnp.mean(jnp.power(predictions - outputs, 2))
    losses.append(loss)
    if kwargs["t"] == kwargs["nsteps"] - 1:
        ax = kwargs["ax"]
        ax.plot(losses, color=colors[agent_name])
        ax.set_title(agent_name.upper())
        plt.tight_layout()
        plt.savefig("asas.png")


def initialize_params(agent_name, **kwargs):
    nfeatures = kwargs["degree"] + 1
    mu0 = jnp.zeros((nfeatures, kwargs["nout"]))
    if agent_name in ["exact bayes", "kf"]:
        mu0 = jnp.zeros((nfeatures, kwargs["nout"]))
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
    nout = 2
    batch_size = 5
    obs_noise = 1.

    env_key, nuts_key, sgld_key = random.split(key, 3)
    env = lambda batch_size: make_random_poly_regression_environment(env_key,
                                                                     degree,
                                                                     ntrain,
                                                                     ntest,
                                                                     nout=nout,
                                                                     obs_noise=obs_noise,
                                                                     train_batch_size=batch_size,
                                                                     test_batch_size=batch_size,
                                                                     x_test_generator=x_test_generator)

    nsteps = 10

    buffer_size = ntrain

    kf = kalman_filter_reg(obs_noise)

    bayes = bayesian_reg(buffer_size, obs_noise)

    optimizer = optax.adam(1e-1)

    nepochs = 4
    sgd = sgd_agent(mean_squared_error,
                    model_fn,
                    optimizer=optimizer,
                    obs_noise=obs_noise,
                    nepochs=nepochs,
                    buffer_size=buffer_size)

    nsamples, nwarmup = 200, 100
    nuts = blackjax_nuts_agent(nuts_key,
                               negative_mean_square_error,
                               model_fn,
                               nsamples=nsamples,
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

    tau = 1.
    strength = obs_noise / tau
    partial_objective_fn = partial(penalized_objective_fn, strength=strength)

    bfgs = bfgs_agent(partial_objective_fn,
                      obs_noise=obs_noise,
                      buffer_size=buffer_size)

    lbfgs = lbfgs_agent(partial_objective_fn,
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
        "sgld": sgld,
        "sgd": sgd,
        "laplace": laplace,
        "bfgs": bfgs,
        "lbfgs": lbfgs,
        "nuts": nuts,
    }
    nrows = len(agents)
    ncols = 1
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
                   timesteps=list(range(nsteps)),
                   nout=nout)


if __name__ == "__main__":
    main()
