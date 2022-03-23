import jax.numpy as jnp
from jax import random

import optax
import flax.linen as nn

from functools import partial
from matplotlib import pyplot as plt

from jsl.experimental.seql.agents.bfgs_agent import bfgs_agent
from jsl.experimental.seql.agents.blackjax_nuts_agent import blackjax_nuts_agent
from jsl.experimental.seql.agents.lbfgs_agent import lbfgs_agent
from jsl.experimental.seql.agents.sgd_agent import sgd_agent
from jsl.experimental.seql.agents.sgmcmc_sgld_agent import sgld_agent
from jsl.experimental.seql.environments.base import make_bimodel_sampler, make_evenly_spaced_x_sampler, make_sin_wave_regression_environment
from jsl.experimental.seql.experiments.plotting import plot_regression_posterior_predictive
from jsl.experimental.seql.utils import mse, train

class MLP(nn.Module):
    nclasses: int

    @nn.compact
    def __call__(self, x
    ):
        x = nn.Dense(5)(x)
        x = nn.tanh(x)
        x = nn.Dense(5)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.nclasses)(x)
        return x

model_fn = None

def logprior_fn(params):
    return 0.

def negative_mean_square_error(params, x, y, model_fn):
    return -mse(params, x, y, model_fn)

def penalized_objective_fn(params, inputs, outputs, model_fn, strength=0.):
    return mse(params, inputs, outputs, model_fn) + strength * jnp.sum(params**2)

def callback_fn(agent_name, env, **kwargs):
    belief = kwargs["belief_state"]

    try:
        mu, sigma = belief.params, None
    except:
        mu, sigma = belief.state.position, None

    ax = next(kwargs["ax"])
    plot_regression_posterior_predictive(ax,
                                        env,
                                        mu,
                                        sigma,
                                        model_fn=model_fn,
                                        obs_noise=kwargs["obs_noise"],
                                        t=kwargs["t"])
    plt.savefig("jaks.png")

def initialize_params(agent_name, **kwargs):

    batch = jnp.ones((1, kwargs["nfeatures"]))
    variables = kwargs["model"].init(kwargs["key"], batch)
    return variables

def sweep(agents, env, train_batch_size, ntrain, nsteps, **init_kwargs):
    nagents = len(agents)

    batch_agents_included = "batch_agents" in init_kwargs
    fig, axes = plt.subplots(nrows=nagents,
                             ncols=nsteps + int(batch_agents_included),
                             figsize=(56, 32))


    for ax, (agent_name, agent) in zip(axes, agents.items()):
        
        params = initialize_params(agent_name, **init_kwargs)
        belief = agent.init_state(params)

        axis_iter = iter(ax)
        partial_callback = lambda **kwargs: callback_fn(agent_name,
                                                        env(train_batch_size),
                                                        ax=axis_iter,
                                                        **init_kwargs,
                                                        **kwargs)

        train(belief, agent, env(train_batch_size),
              nsteps=nsteps, callback=partial_callback)

        if batch_agents_included:
            batch_agent = init_kwargs["batch_agents"][agent_name]
            partial_callback = lambda **kwargs: callback_fn(agent_name,
                                                env(ntrain),
                                                ax=axis_iter,
                                                **init_kwargs,
                                                **kwargs)
            train(belief, batch_agent, env(ntrain),
                nsteps=1, callback=partial_callback)
    plt.savefig("ajsk.png")

def main():
    global model_fn
    key = random.PRNGKey(0)
    
    min_val, max_val = -3, 3

    noutputs = 1
    model = MLP(noutputs)
    model_fn = model.apply

    x_train_generator = make_bimodel_sampler(mixing_parameter=0.5,
                                             means=[-3, 3],
                                             variances=[0.01, 0.01])

    x_test_generator = make_evenly_spaced_x_sampler(max_val,
                                                    use_bias=False,
                                                    min_val=min_val)

    ntrain = 320
    ntest = 64
    train_batch_size = 4

    env_key, nuts_key, sgld_key, model_key = random.split(key, 4)

    obs_noise = 0.01
    env = lambda batch_size: make_sin_wave_regression_environment(env_key,
                                                                    ntrain,
                                                                    ntest,
                                                                    train_batch_size=batch_size,
                                                                    obs_noise=obs_noise,
                                                                    x_train_generator=x_train_generator,
                                                                    x_test_generator=x_test_generator)
                     
    nsteps = 20

    buffer_size = ntrain

    optimizer = optax.adam(1e-4)

    nepochs = 20
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
                        buffer_size=ntrain,
                        nepochs=nepochs*nsteps)

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
                                nsamples=nsamples*nsteps,
                                nwarmup=nwarmup,
                                obs_noise=obs_noise,
                                buffer_size=ntrain)

    partial_logprob_fn = partial(negative_mean_square_error,
                                 model_fn=model_fn)
    dt = 1e-5
    sgld = sgld_agent(sgld_key,
                    partial_logprob_fn,
                    logprior_fn,
                    model_fn,
                    dt = dt,
                    batch_size=train_batch_size,
                    nsamples=nsamples,
                    obs_noise=obs_noise,
                    buffer_size=buffer_size)

    dt = 1e-5
    batch_sgld = sgld_agent(sgld_key,
                            partial_logprob_fn,
                            logprior_fn,
                            model_fn,
                            dt = dt,
                            batch_size=train_batch_size,
                            nsamples=nsamples*nsteps,
                            obs_noise=obs_noise,
                            buffer_size=ntrain)



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

    agents = {
              "sgd": sgd,
              "nuts": nuts,
              "sgld":sgld,
              "bfgs": bfgs,
              "lbfgs": lbfgs,
              }
    
    batch_agents = {
                    "sgd": batch_sgd,
                    "nuts": batch_nuts,
                    "sgld": batch_sgld,
                    "bfgs": batch_bfgs,
                    "lbfgs": batch_lbfgs
                    }
        
    sweep(agents, env, train_batch_size,
          ntrain, nsteps,
          nfeatures=2,
          model=model,
          key=model_key,
          obs_noise=obs_noise,
          batch_agents=batch_agents)


if __name__ == "__main__":
    main()