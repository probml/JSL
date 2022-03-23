import jax.numpy as jnp
from jax import random

from jsl.experimental.seql.agents.blackjax_nuts_agent import blackjax_nuts_agent
from jsl.experimental.seql.environments.base import make_evenly_spaced_x_sampler, make_random_poly_regression_environment
from jsl.experimental.seql.experiments.plotting import plot_regression_posterior_predictive
from jsl.experimental.seql.experiments.experiment_utils import MLP
from jsl.experimental.seql.utils import train, mse


belief = None

def model_fn(w, x):
    return x @ w

def logprob_fn(params, x, y, model_fn):
    return -mse(params, x, y, model_fn)

def callback_fn(env, obs_noise, timesteps, **kwargs):
    global belief
    belief = kwargs["belief_state"]
    mu, sigma = belief.state.position, None
    filename = "poly_reg_nuts_ppd"

    plot_regression_posterior_predictive(env,
                            mu,
                            sigma,
                            obs_noise,
                            timesteps,
                            filename,
                            model_fn=model_fn,
                            **kwargs)


def main():

    key = random.PRNGKey(0)
    degree = 3
    ntrain = 2048  
    ntest = 64
    
    min_val, max_val = -3, 3
    x_test_generator = make_evenly_spaced_x_sampler(max_val,
                                                    use_bias=False,
                                                    min_val=min_val)

    train_batch_size = 128
    env = make_random_poly_regression_environment(key,
                                                  degree,
                                                  ntrain,
                                                  ntest,
                                                  train_batch_size=train_batch_size,
                                                  x_test_generator=x_test_generator)
                                                    
    obs_noise = 0.01
    timesteps = [5, 10, 15]
    nsteps = 22
    buffer_size = train_batch_size
    nsamples, nwarmup = 100, 50

    agent = blackjax_nuts_agent(key,
                                logprob_fn,
                                model_fn,
                                nsamples=nsamples,
                                nwarmup=nwarmup,
                                obs_noise=obs_noise,
                                buffer_size=buffer_size)

    params =  jnp.zeros((degree+1, 1))
    belief = agent.init_state(params)

    partial_callback = lambda **kwargs: callback_fn(env, obs_noise, timesteps, **kwargs)
    
    _ = train(belief,
                              agent,
                              env,
                              nsteps=nsteps,
                              callback=partial_callback)

if __name__ == "__main__":
    main()