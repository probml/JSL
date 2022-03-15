"""Tests for jsl.sent.agents.bayesian_linear_regression"""
import jax.numpy as jnp

from absl.testing import absltest
from jsl.seql.agents.bayesian_lin_reg_agent import bayesian_reg

from jsl.seql.utils import train
from jsl.seql.agents.kf_agent import kalman_filter_reg
from jsl.seql.experiments.linreg_kf_demo import make_matlab_demo_environment


kf_mean, kf_cov = None, None


def callback_fn(**kwargs):
    global kf_mean, kf_cov

    mu_hist = kwargs["info"].mu_hist
    Sigma_hist = kwargs["info"].Sigma_hist

    if kf_mean is not None:
        kf_mean = jnp.vstack([kf_mean, mu_hist])
        kf_cov = jnp.vstack([kf_cov, Sigma_hist])
    else:
        kf_mean = mu_hist
        kf_cov = Sigma_hist


bayes_mean, bayes_cov = None, None


def bayes_callback_fn(**kwargs):
    global bayes_mean, bayes_cov

    mu = kwargs["belief_state"].mu[None, ...]
    Sigma = kwargs["belief_state"].Sigma[None, ...]

    if bayes_mean is not None:
        bayes_mean = jnp.vstack([bayes_mean, mu])
        bayes_cov = jnp.vstack([bayes_cov, Sigma])
    else:
        bayes_mean = mu
        bayes_cov = Sigma


class BayesLinRegTest(absltest.TestCase):

    def test_kf_vs_bayes_on_matlab_demo(self):
        env = make_matlab_demo_environment(test_batch_size=1)

        *_, input_dim = env.X_train.shape

        nsteps = 1
        mu0 = jnp.zeros(input_dim)
        Sigma0 = jnp.eye(input_dim) * 10.

        obs_noise = 2.
        agent = kalman_filter_reg(obs_noise, return_history=True)
        belief = agent.init_state(mu0, Sigma0)

        kf_belief, _ = train(belief, agent, env,
                                  nsteps=nsteps, callback=callback_fn)

        buffer_size = jnp.inf
        agent = bayesian_reg(buffer_size, obs_noise)
        belief = agent.init_state(mu0.reshape((-1, 1)), Sigma0)

        bayes_belief, _ = train(belief, agent, env,
                                  nsteps=nsteps, callback=bayes_callback_fn)

        assert jnp.allclose(jnp.squeeze(kf_belief.mu), jnp.squeeze(bayes_belief.mu))
        assert jnp.allclose(kf_belief.Sigma, bayes_belief.Sigma)

if __name__ == '__main__':
    absltest.main()
