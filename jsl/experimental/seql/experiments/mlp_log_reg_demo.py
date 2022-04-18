import chex
import jax.numpy as jnp
from jax import jit, random, tree_leaves, tree_map
from jax.flatten_util import ravel_pytree

import optax
import haiku as hk
from jaxopt import ScipyMinimize

from functools import partial
from matplotlib import pyplot as plt

from jsl.experimental.seql.agents.bfgs_agent import BFGSAgent
from jsl.experimental.seql.agents.blackjax_nuts_agent import BlackJaxNutsAgent
from jsl.experimental.seql.agents.laplace_agent import LaplaceAgent
from jsl.experimental.seql.agents.lbfgs_agent import LBFGSAgent
from jsl.experimental.seql.agents.sgd_agent import SGDAgent
from jsl.experimental.seql.agents.sgmcmc_sgld_agent import SGLDAgent
from jsl.experimental.seql.agents.eekf_agent import EEKFAgent
from jsl.experimental.seql.environments.base import make_classification_mlp_environment, make_mlp
from jsl.experimental.seql.experiments.experiment_utils import run_experiment
from jsl.experimental.seql.utils import mse, train
from jsl.experimental.seql.experiments.plotting import colors
from jsl.nlds.extended_kalman_filter import NLDS

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
        print(losses)
        ax = kwargs["ax"]
        ax.plot(losses, color=colors[agent_name])
        ax.set_title(agent_name.upper())
        plt.tight_layout()
        plt.savefig("asas.png")


def initialize_params(agent_name, **kwargs):
    nfeatures = kwargs["nfeatures"]
    transformed = kwargs["transformed"]
    key = kwargs["init_key"]

    dummy_input = jnp.zeros([1, nfeatures])
    params = transformed.init(key, dummy_input)

    if agent_name=="eekf":
        mu, _ = ravel_pytree(params)
        sigma_key, key = random.split(key, 2)
        n = mu.size
        Sigma = random.normal(sigma_key,
                              shape=(n, n))
        return (mu, Sigma)

    return (params,)


def sweep(agents, env, train_batch_size, ntrain, nsteps, figsize=(12, 6), **init_kwargs):
    batch_agents_included = "batch_agents" in init_kwargs

    nrows = len(agents)
    ncols = len(init_kwargs["timesteps"]) + int(batch_agents_included)
    fig, big_axes = plt.subplots(nrows=1,
                                 ncols=nrows,
                                 figsize=figsize)

    for idx, (big_ax, (agent_name, agent)) in enumerate(zip(big_axes, agents.items())):
        params = initialize_params(agent_name, **init_kwargs)
        belief = agent.init_state(params)

        partial_callback = lambda **kwargs: callback_fn(agent,
                                                        env(train_batch_size),
                                                        agent_name,
                                                        fig=fig,
                                                        nrows=nrows,
                                                        ncols=ncols,
                                                        idx=idx,
                                                        ax=big_ax,
                                                        **init_kwargs,
                                                        **kwargs)

        train(belief, agent, env(train_batch_size),
              nsteps=nsteps, callback=partial_callback)

    plt.savefig("ajsk.png")


def main():
    key = random.PRNGKey(0)
    model_key, env_key, nuts_key, sgld_key, init_key, eekf_key = random.split(key, 6)
    degree = 3
    ntrain = 50
    ntest = 50
    batch_size = 5
    obs_noise = 1.
    hidden_layer_sizes = [5, 5]
    nfeatures = 100
    ntargets = 2
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

    env = lambda batch_size: make_classification_mlp_environment(env_key,
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
    sgd = SGDAgent(mse,
                   model_fn,
                   optimizer=optimizer,
                   obs_noise=obs_noise,
                   nepochs=nepochs,
                   buffer_size=buffer_size,
                   is_classifier=True)

    batch_sgd = SGDAgent(mse,
                         model_fn,
                         optimizer=optimizer,
                         obs_noise=obs_noise,
                         buffer_size=buffer_size,
                         nepochs=nepochs * nsteps,
                         is_classifier=True)

    nsamples, nwarmup = 200, 100
    nuts = BlackJaxNutsAgent(
        negative_mean_square_error,
        model_fn,
        nsamples=nsamples,
        nwarmup=nwarmup,
        obs_noise=obs_noise,
        buffer_size=buffer_size,
        is_classifier=True)

    batch_nuts = BlackJaxNutsAgent(negative_mean_square_error,
                                   model_fn,
                                   nsamples=nsamples * nsteps,
                                   nwarmup=nwarmup,
                                   obs_noise=obs_noise,
                                   buffer_size=buffer_size,
                                   is_classifier=True)

    partial_logprob_fn = partial(negative_mean_square_error,
                                 model_fn=model_fn)
    dt = 1e-4
    sgld = SGLDAgent(
        partial_logprob_fn,
        logprior_fn,
        model_fn,
        dt=dt,
        batch_size=batch_size,
        nsamples=nsamples,
        obs_noise=obs_noise,
        buffer_size=buffer_size,
        is_classifier=True)

    dt = 1e-5
    batch_sgld = SGLDAgent(
        partial_logprob_fn,
        logprior_fn,
        model_fn,
        dt=dt,
        batch_size=batch_size,
        nsamples=nsamples * nsteps,
        obs_noise=obs_noise,
        buffer_size=buffer_size,
        is_classifier=True)

    tau = 1.
    strength = obs_noise / tau
    partial_objective_fn = partial(penalized_objective_fn, strength=strength)

    bfgs = BFGSAgent(partial_objective_fn,
                     model_fn=model_fn,
                     obs_noise=obs_noise,
                     buffer_size=buffer_size,
                     is_classifier=True)

    batch_bfgs = BFGSAgent(partial_objective_fn,
                           model_fn=model_fn,
                           obs_noise=obs_noise,
                           buffer_size=buffer_size,
                           is_classifier=True)

    lbfgs = LBFGSAgent(partial_objective_fn,
                       model_fn=model_fn,
                       obs_noise=obs_noise,
                       history_size=buffer_size,
                       is_classifier=True)

    batch_lbfgs = LBFGSAgent(partial_objective_fn,
                             model_fn=model_fn,
                             obs_noise=obs_noise,
                             history_size=buffer_size,
                             is_classifier=True)

    energy_fn = partial(partial_objective_fn, model_fn=model_fn)
    solver = ScipyMinimize(fun=energy_fn, method="BFGS")
    laplace = LaplaceAgent(solver,
                           energy_fn,
                           model_fn,
                           obs_noise=obs_noise,
                           buffer_size=buffer_size,
                           is_classifier=True)

    initial_params = transformed.init(eekf_key,
                                      jnp.zeros((ntrain, nfeatures)))
    flat_params, unflatten_fn = ravel_pytree(initial_params)

    def fx(flat_params, x):
        params = unflatten_fn(flat_params)
        return model_fn(params, x)

    Q = jnp.eye(flat_params.size) * 1e-4  # parameters do not change
    R = jnp.eye(1) * obs_noise
    nlds = NLDS(lambda x: x, fx, Q, R)
    eekf = EEKFAgent(nlds, is_classifier=True)

    agents = {
        "eekf": eekf,
        "sgd": sgd,
        "laplace": laplace,
        "bfgs": bfgs,
        "lbfgs": lbfgs,
        "nuts": nuts,
        "sgld": sgld,
    }

    batch_agents = {
        "eekf": eekf,
        "sgd": batch_sgd,
        "laplace": laplace,
        "bfgs": batch_bfgs,
        "lbfgs": batch_lbfgs,
        "nuts": batch_nuts,
        "sgld": batch_sgld,
    }

    nrows = len(agents)
    ncols = 1

    nsamples, njoint = 20, 10

    run_experiment(init_key,
                   agents,
                   env,
                   initialize_params,
                   batch_size,
                   ntrain,
                   nsteps,
                   nsamples,
                   njoint,
                   nrows,
                   ncols,
                   callback_fn,
                   degree=degree,
                   obs_noise=obs_noise,
                   timesteps=list(range(nsteps)),
                   init_key=init_key,
                   nfeatures=nfeatures,
                   transformed=transformed,
                   batch_agents=batch_agents)


if __name__ == "__main__":
    main()
