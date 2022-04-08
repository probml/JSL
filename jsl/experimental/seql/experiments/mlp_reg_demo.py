import chex
import jax.numpy as jnp
from jax import jit, random, tree_leaves, tree_map

import optax
import haiku as hk
from jaxopt import ScipyMinimize

from functools import partial
from matplotlib import pyplot as plt

from jsl.experimental.seql.agents.bfgs_agent import bfgs_agent
from jsl.experimental.seql.agents.blackjax_nuts_agent import blackjax_nuts_agent
from jsl.experimental.seql.agents.laplace_agent import laplace_agent
from jsl.experimental.seql.agents.lbfgs_agent import lbfgs_agent
from jsl.experimental.seql.agents.sgd_agent import sgd_agent
from jsl.experimental.seql.agents.sgmcmc_sgld_agent import sgld_agent
from jsl.experimental.seql.environments.base import make_mlp, make_regression_mlp_environment
from jsl.experimental.seql.experiments.experiment_utils import run_experiment
from jsl.experimental.seql.utils import mse, train
from jsl.experimental.seql.experiments.plotting import colors

plt.style.use("seaborn-poster")


def logprior_fn(params):
    return sum(tree_leaves(tree_map(lambda x: jnp.sum(x ** 2), params)))


def negative_mean_square_error(params, inputs, outputs, model_fn, strength=0.2):
    return -penalized_objective_fn(params, inputs, outputs, model_fn, strength=strength)


def penalized_objective_fn(params, inputs, outputs, model_fn, strength=0.2):
    return mse(params, inputs, outputs, model_fn) + strength * logprior_fn(params) / len(inputs)


def energy_fn(params, data, model_fn, strength=0.2):
    return mse(params, *data, model_fn) + strength * logprior_fn(params) / len(data[0])


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
    nfeatures = kwargs["nfeatures"]
    transformed = kwargs["transformed"]
    key = kwargs["key"]

    dummy_input = jnp.zeros([1, nfeatures])
    params = transformed.init(key, dummy_input)
    return params


def main():
    key = random.PRNGKey(0)
    model_key, env_key, nuts_key, sgld_key, init_key = random.split(key, 5)
    degree = 3
    ntrain = 50
    ntest = 50
    batch_size = 5
    obs_noise = 1.
    hidden_layer_sizes = [5, 5]
    nfeatures = 100
    ntargets = 1
    temperature = 1.

    net_fn = make_mlp(model_key,
                      nfeatures,
                      ntargets,
                      temperature,
                      hidden_layer_sizes)

    transformed = hk.without_apply_rng(hk.transform(net_fn))

    assert temperature > 0.0

    def forward(params: chex.Array, x: chex.Array):
        return transformed.apply(params, x) / temperature

    model_fn = jit(forward)

    env = lambda batch_size: make_regression_mlp_environment(env_key,
                                                             nfeatures,
                                                             ntargets,
                                                             ntrain,
                                                             ntest,
                                                             temperature=1.,
                                                             hidden_layer_sizes=hidden_layer_sizes,
                                                             train_batch_size=batch_size,
                                                             test_batch_size=batch_size,
                                                             )

    nsteps = 10

    buffer_size = ntrain

    optimizer = optax.adam(1e-3)

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
        "nuts": nuts,
        "sgld": sgld,
        "sgd": sgd,
        "laplace": laplace,
        "bfgs": bfgs,
        "lbfgs": lbfgs,
    }

    batch_agents = {
        "sgd": batch_sgd,
        "laplace": laplace,
        "bfgs": batch_bfgs,
        "lbfgs": batch_lbfgs,
        "nuts": batch_nuts,
        "sgld": batch_sgld,
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
                   key=init_key,
                   nfeatures=nfeatures,
                   transformed=transformed,
                   batch_agents=batch_agents)


if __name__ == "__main__":
    main()
