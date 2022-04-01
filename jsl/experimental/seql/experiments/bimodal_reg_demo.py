import jax.numpy as jnp
from jax import random, tree_map
from jax.tree_util import tree_leaves

import optax
from jaxopt import ScipyMinimize
import flax.linen as nn

from functools import partial
from matplotlib import pyplot as plt

from jsl.experimental.seql.agents.bfgs_agent import bfgs_agent
from jsl.experimental.seql.agents.blackjax_nuts_agent import blackjax_nuts_agent
from jsl.experimental.seql.agents.laplace_agent import laplace_agent
from jsl.experimental.seql.agents.lbfgs_agent import lbfgs_agent
from jsl.experimental.seql.agents.lbfgsb_agent import lbfgsb_agent
from jsl.experimental.seql.agents.sgd_agent import sgd_agent
from jsl.experimental.seql.agents.sgmcmc_sgld_agent import sgld_agent
from jsl.experimental.seql.environments.base import make_evenly_spaced_x_sampler, make_sin_wave_regression_environment
from jsl.experimental.seql.experiments.plotting import plot_regression_posterior_predictive
from jsl.experimental.seql.utils import mse, train


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

def logprior_fn(params, strength=0.2):
    leaves = tree_leaves(params)
    return -sum(tree_map(lambda x : jnp.sum(x**2), leaves)) * strength

def loglikelihood_fn(params, x, y, model_fn):
    return -mse(params, x, y, model_fn)*len(x)

def logjoint_fn(params, x, y, model_fn, strength=0.2):
    return loglikelihood_fn(params, x, y, model_fn) + logprior_fn(params, strength=strength)

def neg_logjoint_fn(params, inputs, outputs, model_fn, strength=0.1):
    return -logjoint_fn(params, inputs, outputs, model_fn, strength=strength)

def callback_fn(agent, env, agent_name, **kwargs):

    if "subplot_idx" not in kwargs and kwargs["t"] not in kwargs["timesteps"]:
        return
    elif "subplot_idx" not in kwargs:
        subplot_idx = kwargs["timesteps"].index(kwargs["t"]) + kwargs["idx"] * kwargs["ncols"] + 1
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
                                        agent_name,
                                        timesteps=kwargs["timesteps"],
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

def sweep(agents, env, train_batch_size, ntrain, nsteps, figsize=(56, 48), **init_kwargs):

    batch_agents_included = "batch_agents" in init_kwargs

    nrows = len(agents)
    ncols = len(init_kwargs["timesteps"]) + int(batch_agents_included)
    fig, big_axes = plt.subplots(nrows=nrows,
                                ncols=1,
                                figsize=figsize)


    for idx, (big_ax, (agent_name, agent)) in enumerate(zip(big_axes, agents.items())):
        
        big_ax.set_title(agent_name.upper(), fontsize=36, y=1.2)
        # Turn off axis lines and ticks of the big subplot 
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), 
                           top='off',
                           bottom='off',
                           left='off',
                           right='off')
        # removes the white frame
        big_ax._frameon = False

        params = initialize_params(agent_name, **init_kwargs)
        belief = agent.init_state(params)

        partial_callback = lambda **kwargs: callback_fn(
                                                        agent,
                                                        env(train_batch_size),
                                                        agent_name,
                                                        fig=fig,
                                                        nrows=nrows,
                                                        ncols=ncols,
                                                        idx=idx,
                                                        **init_kwargs,
                                                        **kwargs)

        train(belief, agent, env(train_batch_size),
              nsteps=nsteps, callback=partial_callback)

        if batch_agents_included:
            batch_agent = init_kwargs["batch_agents"][agent_name]
            partial_callback = lambda **kwargs: callback_fn(batch_agent,
                                                env(ntrain),
                                                agent_name,
                                                fig=fig,
                                                nrows=nrows,
                                                ncols=ncols,
                                                idx=idx,
                                                title="Batch Agent",
                                                subplot_idx=(idx+1) * ncols,
                                                **init_kwargs,
                                                **kwargs)

            train(belief, batch_agent, env(ntrain),
                nsteps=1, callback=partial_callback)

def main():
    key = random.PRNGKey(0)
    
    min_val, max_val = -10, 10

    noutputs = 1

    '''x_train_generator = make_bimodel_sampler(mixing_parameter=0.5,
                                             means=[-2, 2],
                                             variances=[0.1, 0.1])'''

    
    x_train_generator = make_evenly_spaced_x_sampler(max_val,
                                                    use_bias=False,
                                                    min_val=min_val)

    x_test_generator = make_evenly_spaced_x_sampler(max_val,
                                                    use_bias=False,
                                                    min_val=min_val)

    ntrain = 50
    ntest = 10
    train_batch_size = 5

    env_key, nuts_key, sgld_key, model_key = random.split(key, 4)

    obs_noise = 0.01
    env = lambda batch_size: make_sin_wave_regression_environment(env_key,
                                                                    ntrain,
                                                                    ntest,
                                                                    train_batch_size=batch_size,
                                                                    obs_noise=obs_noise,
                                                                    x_train_generator=x_train_generator,
                                                                    x_test_generator=x_test_generator)
                     
    nsteps = 10

    buffer_size = ntrain

    optimizer = optax.adam(1e-1)

    nepochs = 40
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

    nsamples, nwarmup = 300, 200
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
                                nsamples=nsamples*nsteps,
                                nwarmup=nwarmup,
                                obs_noise=obs_noise,
                                buffer_size=ntrain)

    partial_logprob_fn = partial(loglikelihood_fn,
                                 model_fn=model_fn)
    dt = 3e-4
    sgld = sgld_agent(sgld_key,
                    partial_logprob_fn,
                    logprior_fn,
                    model_fn,
                    dt = dt,
                    batch_size=-1,
                    nsamples=nsamples+nwarmup,
                    nlast=nsamples,
                    obs_noise=obs_noise,
                    buffer_size=buffer_size)

    dt = 3e-4
    batch_sgld = sgld_agent(sgld_key,
                            partial_logprob_fn,
                            logprior_fn,
                            model_fn,
                            dt = dt,
                            batch_size=train_batch_size,
                            nsamples=(nsamples + nwarmup)*nsteps,
                            nlast = nsamples*nsteps,
                            obs_noise=obs_noise,
                            buffer_size=ntrain)



    tau = 1.
    strength = obs_noise / tau
    partial_objective_fn = partial(neg_logjoint_fn, strength=strength)

    bfgs = bfgs_agent(partial_objective_fn,
                      model_fn=model_fn,
                      obs_noise=obs_noise,
                      buffer_size=buffer_size)

    lbfgs = lbfgsb_agent(partial_objective_fn,
                      model_fn=model_fn,
                      obs_noise=obs_noise,
                      buffer_size=buffer_size)

    energy_fn = partial(partial_objective_fn, model_fn=model_fn)
    solver = ScipyMinimize(fun=energy_fn, method="BFGS")
    laplace = laplace_agent(solver,
                            energy_fn,
                            model_fn,
                            obs_noise=obs_noise,
                            buffer_size=buffer_size)


    agents = {
              "bfgs": bfgs,
              "lbfgs": lbfgs,
              "sgd": sgd,
              "nuts":nuts,
              "sgld":sgld,
              }
    
    batch_agents = {
                    "laplace": laplace,
                    "sgd": batch_sgd,
                    "nuts": batch_nuts,
                    "sgld": batch_sgld,
                    "bfgs": bfgs,
                    "lbfgs": lbfgs
                    }
        
    sweep(agents, env, train_batch_size,
          ntrain, nsteps,
          nfeatures=2,
          model=model,
          key=model_key,
          obs_noise=obs_noise,
          timesteps=list(range(nsteps)),
          batch_agents=batch_agents)


if __name__ == "__main__":
    main()