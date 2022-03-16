import jax.numpy as jnp
from jax import random

from jsl.experimental.seql.agents.kf_agent import kalman_filter_reg
from jsl.experimental.seql.environments.base import make_evenly_spaced_x_sampler, make_random_poly_regression_environment
from jsl.experimental.seql.experiments.plotting import plot_posterior_predictive
from jsl.experimental.seql.utils import train


belief = None

def callback_fn(env, obs_noise, timesteps, **kwargs):
    global belief
    belief = kwargs["belief_state"]
    mu, sigma = belief.mu, belief.Sigma
    filename = "poly_reg_kf_ppd"
    plot_posterior_predictive(env,
                              mu,
                              sigma,
                              obs_noise,
                              timesteps,
                              filename,
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
    agent = kalman_filter_reg(obs_noise)

    input_dim = env.X_train.shape[-1]
    mu0 = jnp.zeros((input_dim,))
    Sigma0 = jnp.eye(input_dim)

    belief = agent.init_state(mu0, Sigma0)

    timesteps = [5, 10, 15]

    partial_callback = lambda **kwargs: callback_fn(env, obs_noise, timesteps, **kwargs)

    nsteps = 20
    _, unused_rewards = train(belief,
                              agent,
                              env,
                              nsteps=nsteps,
                              callback=partial_callback)

if __name__ == "__main__":
    main()