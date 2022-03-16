import jax.numpy as jnp
from jax import random

from functools import partial

from jsl.experimental.seql.agents.sgd_agent import sgd_agent
from jsl.experimental.seql.environments.base import make_evenly_spaced_x_sampler, make_random_poly_regression_environment
from jsl.experimental.seql.experiments.plotting import plot_posterior_predictive
from jsl.experimental.seql.experiments.experiment_utils import MLP
from jsl.experimental.seql.utils import train, mse


belief = None

def callback_fn(model, env, obs_noise, timesteps, **kwargs):
    global belief
    belief = kwargs["belief_state"]
    mu, sigma = belief.params, None
    filename = "poly_reg_sgd_ppd"

    plot_posterior_predictive(env,
                            mu,
                            sigma,
                            obs_noise,
                            timesteps,
                            filename,
                            model_fn=model.apply,
                            **kwargs)


def main():

    key = random.PRNGKey(0)
    degree = 3
    ntrain = 200  # 80% of the data
    ntest = 50  # 20% of the data
    
    min_val, max_val = -3, 3
    x_test_generator = make_evenly_spaced_x_sampler(max_val,
                                                    use_bias=False,
                                                    min_val=min_val)

    env = make_random_poly_regression_environment(key,
                                                  degree,
                                                  ntrain,
                                                  ntest,
                                                  x_test_generator=x_test_generator)
                                                    
    obs_noise = 0.01
    timesteps = [5, 10, 15]
    nsteps = 20
    buffer_size = 1
    nclasses = 1
    model = MLP(nclasses)

    partial_mse = partial(mse, model_fn=model.apply)
    agent = sgd_agent(partial_mse,
                      model.apply,
                      obs_noise=obs_noise,
                      buffer_size=buffer_size)

    batch = jnp.ones((1, degree + 1))
    variables = model.init(key, batch)
    belief = agent.init_state(variables)

    partial_callback = lambda **kwargs: callback_fn(model, env, obs_noise, timesteps, **kwargs)
    
    _, unused_rewards = train(belief,
                              agent,
                              env,
                              nsteps=nsteps,
                              callback=partial_callback)

if __name__ == "__main__":
    main()