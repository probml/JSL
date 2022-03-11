"""Tests for jsl.sent.agents.eekf"""
import jax.numpy as jnp
from jax.nn import sigmoid

from absl.testing import absltest

from jsl.nlds.base import NLDS
from jsl.nlds.extended_kalman_filter import filter
from jsl.seql.train import train
from jsl.seql.experiments.logreg_eekf_demo import make_biclusters_data_environment
from jsl.seql.agents.eekf import eekf


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

def fz(x): return x
def fx(w, x): return sigmoid(w[None, :] @ x)
def Rt(w, x): return (sigmoid(w @ x) * (1 - sigmoid(w @ x)))[None, None]

class EEKFTest(absltest.TestCase):

  def test_eekf(self):
      train_batch_size = 1
      test_batch_size = 1
      env = make_biclusters_data_environment(train_batch_size,
                                              test_batch_size)
                                              
  
      Phi = jnp.squeeze(env.X_train)
      y = jnp.squeeze(env.y_train)
      n_datapoints, input_dim =  Phi.shape

      mu_t = jnp.zeros(input_dim)
      Pt = jnp.eye(input_dim) * 0.0
      P0 = jnp.eye(input_dim) * 2.0

      ### EEKF Approximation
      nlds = NLDS(fz, fx, Pt, Rt, mu_t, P0)
      agent = eekf(nlds)
      belief = agent.init_state(mu_t, P0)
      unused_rewards = train(belief, agent, env, n_datapoints, callback_fn)

      w_eekf_hist = mean
      P_eekf_hist = cov

      _, eekf_hist = filter(nlds, mu_t, y,
                                                     Phi, P0,
                                                     return_params=["mean", "cov"])

      assert jnp.allclose(w_eekf_hist, eekf_hist["mean"])
      assert jnp.allclose(P_eekf_hist, eekf_hist["cov"])

if __name__ == '__main__':
  absltest.main()