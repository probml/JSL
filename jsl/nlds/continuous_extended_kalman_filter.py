import jax
import jax.numpy as jnp
from jax.ops import index_update
from math import ceil

class ContinuousExtendedKalmanFilter:
    """
    Extended Kalman Filter for a nonlinear continuous time
    dynamical system with observations in discrete time.
    """
    def __init__(self, fz, fx, Q, R):
        self.fz = fz
        self.fx = fx
        self.Dfz = jax.jacrev(fz)
        self.Dfx = jax.jacrev(fx)
        self.Q = Q
        self.R = R
        self.state_size, _ = Q.shape
        self.obs_size, _ = R.shape
        
    @staticmethod
    def _rk2(x0, f, nsteps, dt):
        """
        class-independent second-order Runge-Kutta method
        
        Parameters
        ----------
        x0: array(state_size, )
            Initial state of the system
        f: function
            Function to integrate. Must return jax.numpy
            array of size state_size
        nsteps: int
            Total number of steps to integrate
        dt: float
            integration step size
        
        Returns
        -------
        array(nsteps, state_size)
            Integration history
        """
        input_dim, *_ = x0.shape
        simulation = jnp.zeros((nsteps, input_dim))
        simulation = index_update(simulation, 0, x0)
        
        xt = x0.copy()
        for t in range(1, nsteps):
            k1 = f(xt)
            k2 = f(xt + dt * k1)
            xt = xt + dt * (k1 + k2) / 2
            simulation = index_update(simulation, t, xt)
        return simulation
    
    def sample(self, key, x0, T, nsamples, dt=0.01, noisy=False):
        """
        Run the Extended Kalman Filter algorithm. First, we integrate
        up to time T, then we obtain nsamples equally-spaced points. Finally,
        we transform the latent space to obtain the observations

        Parameters
        ----------
        key: jax.random.PRNGKey
            Initial seed
        x0: array(state_size)
            Initial state of simulation
        T: float
            Final time of integration
        nsamples: int
            Number of observations to take from the total integration
        dt: float
            integration step size
        noisy: bool
            Whether to (naively) add noise to the state space

        Returns
        -------
        * array(nsamples, state_size)
            State-space values
        * array(nsamples, obs_size)
            Observed-space values
        * int
            Number of observations skipped between one
            datapoint and the next
        """
        nsteps = ceil(T / dt)
        jump_size = ceil(nsteps / nsamples)
        correction = nsamples - ceil(nsteps / jump_size)
        nsteps += correction * jump_size

        key_state, key_obs = jax.random.split(key)
        state_noise = jax.random.multivariate_normal(key_state, jnp.zeros(self.state_size), self.Q, (nsteps,))
        obs_noise = jax.random.multivariate_normal(key_obs, jnp.zeros(self.obs_size), self.R, (nsteps,)) 
        simulation = self._rk2(x0, self.fz, nsteps, dt)
        
        if noisy:
            simulation = simulation + jnp.sqrt(dt) * state_noise
        
        sample_state = simulation[::jump_size]
        sample_obs = jnp.apply_along_axis(self.fx, 1, sample_state) + obs_noise[:len(sample_state)]
        
        return sample_state, sample_obs, jump_size
    
    def _Vt_dot(self, V, G):
        return G @ V @ G.T + self.Q
    
    def estimate(self, sample_state, sample_obs, jump_size, dt):
        """
        Run the Extended Kalman Filter algorithm over a set of observed samples.

        Parameters
        ----------
        sample_state: array(nsamples, state_size)
        sample_obs: array(nsamples, obs_size)
        jump_size: int
        dt: float

        Returns
        -------
        * array(nsamples, state_size)
            History of filtered mean terms
        * array(nsamples, state_size, state_size)
            History of filtered covariance terms
        """
        I = jnp.eye(self.state_size)
        nsamples = len(sample_state)
        Vt = self.R.copy()
        mu_t = sample_state[0]

        mu_hist = jnp.zeros((nsamples, self.state_size))
        V_hist = jnp.zeros((nsamples, self.state_size, self.state_size))

        mu_hist = index_update(mu_hist, 0, mu_t)
        V_hist = index_update(V_hist, 0, Vt)

        for t in range(1, nsamples):
            for _ in range(jump_size):
                k1 = self.fz(mu_t)
                k2 = self.fz(mu_t + dt * k1)
                mu_t = mu_t + dt * (k1 + k2) / 2

                Gt = self.Dfz(mu_t)
                k1 = self._Vt_dot(Vt, Gt)
                k2 = self._Vt_dot(Vt + dt * k1, Gt)
                Vt = Vt + dt * (k1 + k2) / 2
            
            mu_t_cond = mu_t
            Vt_cond = Vt
            Ht = self.Dfx(mu_t_cond)

            Kt = Vt_cond @ Ht.T @ jnp.linalg.inv(Ht @ Vt_cond @ Ht.T + self.R)
            mu_t = mu_t_cond + Kt @ (sample_obs[t] - self.fx(mu_t_cond))
            Vt = (I - Kt @ Ht) @ Vt_cond

            mu_hist = index_update(mu_hist, t, mu_t)
            V_hist = index_update(V_hist, t, Vt)
        
        return mu_hist, V_hist
