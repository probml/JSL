import jax.numpy as jnp
from jax import random

from functools import partial
from matplotlib import pyplot as plt
import optax
from jsl.experimental.seql.agents.bayesian_lin_reg_agent import bayesian_reg


from jsl.experimental.seql.agents.bfgs_agent import bfgs_agent
from jsl.experimental.seql.agents.blackjax_nuts_agent import blackjax_nuts_agent
from jsl.experimental.seql.agents.kf_agent import kalman_filter_reg
from jsl.experimental.seql.agents.lbfgs_agent import lbfgs_agent
from jsl.experimental.seql.agents.sgd_agent import sgd_agent
from jsl.experimental.seql.agents.sgmcmc_sgld_agent import sgld_agent
from jsl.experimental.seql.environments.base import make_evenly_spaced_x_sampler, make_random_poly_regression_environment
from jsl.experimental.seql.experiments.plotting import plot_posterior_predictive
from jsl.experimental.seql.utils import mse, train
    

belief = None


def model_fn(w, x):
    return x @ w

def logprior_fn(params):
    return 0.

def negative_mean_square_error(params, x, y, model_fn):
    return -mse(params, x, y, model_fn)

def penalized_objective_fn(params, inputs, outputs, model_fn, strength=0.):
    return mse(params, inputs, outputs, model_fn) + strength * jnp.sum(params**2)

def callback_fn(agent_name, env, **kwargs):
    global belief
    belief = kwargs["belief_state"]
    
    try:
        mu, sigma = belief.mu, belief.Sigma
    except:

        try:
            mu, sigma = belief.params, None
        except:
            mu, sigma = belief.state.position, None
    ax = next(kwargs["ax"])
    plot_posterior_predictive(ax,
                              env,
                              mu,
                              sigma,
                              model_fn=model_fn,
                              obs_noise=kwargs["obs_noise"],
                              t=kwargs["t"])
    plt.savefig("jaks.png")

def initialize_params(agent_name, **kwargs):
    nfeatures = kwargs["degree"] + 1
    mu0 = jnp.zeros((nfeatures, 1))
    if agent_name == "bayes" or  agent_name == "kf":
        mu0 = jnp.zeros((nfeatures, 1))
        Sigma0 = jnp.eye(nfeatures)
        initial_params = (mu0, Sigma0)
    else:
        initial_params = (mu0,)

    return initial_params

def sweep(agents, env, train_batch_size, ntrain, nsteps, **init_kwargs):
    nagents = len(agents)

    batch_agents_included = "batch_agents" in init_kwargs
    fig, axes = plt.subplots(nrows=nagents,
                             ncols=nsteps + int(batch_agents_included),
                             figsize=(56, 32))


    for ax, (agent_name, agent) in zip(axes, agents.items()):
        
        params = initialize_params(agent_name, **init_kwargs)
        belief = agent.init_state(*params)

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

    key = random.PRNGKey(0)
    
    min_val, max_val = -3, 3
    x_test_generator = make_evenly_spaced_x_sampler(max_val,
                                                    use_bias=False,
                                                    min_val=min_val)

    degree = 3
    ntrain = 2048  
    ntest = 64
    train_batch_size = 4

    env_key, nuts_key, sgld_key = random.split(key, 3)
    env = lambda batch_size: make_random_poly_regression_environment(env_key,
                                                  degree,
                                                  ntrain,
                                                  ntest,
                                                  train_batch_size=batch_size,
                                                  x_test_generator=x_test_generator)
                                                    
    buffer_size = 20
    obs_noise = 0.01
    nsteps = 12

    buffer_size = train_batch_size

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
                        buffer_size=ntrain,
                        nepochs=nepochs*nsteps)

    nsamples, nwarmup = 100, 50
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
                    buffer_size=ntrain)

    lbfgs = lbfgs_agent(partial_objective_fn,
                        obs_noise=obs_noise,
                        buffer_size=buffer_size)

    batch_lbfgs = lbfgs_agent(partial_objective_fn,
                    obs_noise=obs_noise,
                    buffer_size=ntrain)

    agents = {
              "kf": kf,
              "bayes": bayes,
              "sgd": sgd,
              "nuts": nuts,
              "sgld":sgld,
              "bfgs": bfgs,
              "lbfgs": lbfgs
              }
    
    batch_agents = {
                    "kf": kf,
                    "bayes": batch_bayes,
                    "sgd": batch_sgd,
                    "nuts": batch_nuts,
                    "sgld": batch_sgld,
                    "bfgs": batch_bfgs,
                    "lbfgs": batch_lbfgs
                    }
        
    sweep(agents, env, train_batch_size,
          ntrain, nsteps,
          degree=degree,
          obs_noise=obs_noise,
          batch_agents=batch_agents)


if __name__ == "__main__":
    main()