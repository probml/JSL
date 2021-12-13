import jax.numpy as jnp
from jax import jacrev
from jax.lax import scan
from .base import NLDS
from functools import partial


class ExtendedKalmanFilter(NLDS):
    """
    Implementation of the Extended Kalman Filter for a nonlinear
    dynamical system with discrete observations
    """

    def __init__(self, fz, fx, Q, R):
        super().__init__(fz, fx, Q, R)
        self.Dfz = jacrev(fz)
        self.Dfx = jacrev(fx)

    @classmethod
    def from_base(cls, model):
        """
        Initialise class from an instance of the NLDS parent class
        """
        return cls(model.fz, model.fx, model.Q, model.R)

    def filter_step(self, state, xs, eps=0.001, carry_params=None):
        """
        Run the Extended Kalman filter algorithm for a single step
        Paramters
        ---------
        state: tuple
            Mean, covariance at time t-1
        xs: tuple
            Target value and observations at time t
        eps: float
            Small number to prevent singular matrix
        carry_params: list
            Fix elements to carry
        """
        mu_t, Vt, t = state
        xt, obs = xs

        state_size, *_ = mu_t.shape
        I = jnp.eye(state_size)
        Gt = self.Dfz(mu_t)
        mu_t_cond = self.fz(mu_t)
        Vt_cond = Gt @ Vt @ Gt.T + self.Q(mu_t, t)
        Ht = self.Dfx(mu_t_cond, *obs)

        Rt = self.R(mu_t_cond, *obs)
        num_inputs, *_ = Rt.shape

        xt_hat = self.fx(mu_t_cond, *obs)
        Mt = Ht @ Vt_cond @ Ht.T + Rt + eps * jnp.eye(num_inputs)
        Kt = Vt_cond @ Ht.T @ jnp.linalg.inv(Mt)
        mu_t = mu_t_cond + Kt @ (xt - xt_hat)
        Vt = (I - Kt @ Ht) @ Vt_cond @ (I - Kt @ Ht).T + Kt @ Rt @ Kt.T
        
        elements = {"mean": mu_t, "cov": Vt}
        return (mu_t, Vt, t + 1), {key: val for key, val in elements.items() if key in carry_params}

    def filter(self, init_state, sample_obs, observations=None, Vinit=None, carry_params=None):
        """
        Run the Extended Kalman Filter algorithm over a set of observed samples.
        Parameters
        ----------
        init_state: array(state_size)
        sample_obs: array(nsamples, obs_size)
        Returns
        -------
        * array(nsamples, state_size)
            History of filtered mean terms
        * array(nsamples, state_size, state_size)
            History of filtered covariance terms
        """
        self.state_size, *_ = init_state.shape

        Vt = self.Q(init_state) if Vinit is None else Vinit

        t = 0
        state = (init_state, Vt, t)
        observations = (observations,) if type(observations) is not tuple else observations
        xs = (sample_obs, observations)

        carry_params = [] if carry_params is None else carry_params
        filter = partial(self.filter_step, carry_params=carry_params)
        (mu_t, Vt, _), hist_elements = scan(filter, state, xs)

        return (mu_t, Vt), hist_elements
