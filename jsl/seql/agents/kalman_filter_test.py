"""Tests for jsl.sent.agents.kalman_filter"""
import jax.numpy as jnp

from absl.testing import absltest

from jsl.seql.train import train
from jsl.seql.agents.kalman_filter import kalman_filter_reg
from jsl.seql.experiments.linreg_kf_demo import make_matlab_demo_environment
from jsl.demos.linreg_kf import kf_linreg

mean, cov = None, None

def callback_fn(**kwargs):
    global mean, cov

    mu_hist = kwargs["info"].mu_hist
    Sigma_hist = kwargs["info"].Sigma_hist

    if mean is not None:
        mean =jnp.vstack([mean, mu_hist])
        cov =jnp.vstack([cov, Sigma_hist])
    else:
        mean = mu_hist
        cov = Sigma_hist

class KalmanFilterTest(absltest.TestCase):

  def test_kalman_filter(self):
    env = make_matlab_demo_environment(test_batch_size=1)

    nsteps, _, input_dim = env.X_train.shape
    
    mu0 = jnp.zeros(input_dim)
    Sigma0 = jnp.eye(input_dim) * 10.

    obs_noise = 1
    agent = kalman_filter_reg(obs_noise)
    belief = agent.init_state(mu0, Sigma0)

    unused_rewards = train(belief, agent, env,
                           nsteps=nsteps, callback=callback_fn)

    w0_hist, w1_hist = mean.T
    w0_err, w1_err = jnp.sqrt(cov[:, [0, 1], [0, 1]].T)

    F = jnp.eye(2)
    mu0 = jnp.zeros(2)
    Sigma0 = jnp.eye(2) * 10.
    Q, R = 0, 1

    X = jnp.squeeze(env.X_train)
    y = jnp.squeeze(env.y_train)
    mu_hist, Sigma_hist = kf_linreg(X, y, R, mu0,
                                    Sigma0, F, Q)

    orig_w0_hist, orig_w1_hist = mu_hist.T
    orig_w0_err, orig_w1_err = jnp.sqrt(Sigma_hist[:,
                                                   [0, 1],
                                                   [0, 1]].T
                                        )

    assert jnp.allclose(w0_hist, orig_w0_hist)
    assert jnp.allclose(w1_hist, orig_w1_hist)
    assert jnp.allclose(w0_err, orig_w0_err)
    assert jnp.allclose(w1_err, orig_w1_err)

if __name__ == '__main__':
  absltest.main()