"""
Extended Kalman Filter for a nonlinear continuous time
dynamical system with observations in discrete time.
"""

import jax
import jax.numpy as jnp
from jax import lax, jacrev

import chex

from math import ceil

from jsl.nlds.base import NLDS


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

    def step(xt, t):
        k1 = f(xt)
        k2 = f(xt + dt * k1)
        xt = xt + dt * (k1 + k2) / 2
        return xt, xt

    steps = jnp.arange(nsteps)
    _, simulation = lax.scan(step, x0, steps)

    simulation = jnp.vstack([x0, simulation])
    return simulation


def sample(key: chex.PRNGKey,
           params: NLDS,
           x0: chex.Array,
           T: float,
           nsamples: int,
           dt: float = 0.01,
           noisy: bool = False):
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

    fz, fx = params.fz, params.fx
    Q, R = params.Qz, params.Rx

    state_size, _ = Q.shape
    obs_size, _ = R.shape

    nsteps = ceil(T / dt)
    jump_size = ceil(nsteps / nsamples)
    correction = nsamples - ceil(nsteps / jump_size)
    nsteps += correction * jump_size

    key_state, key_obs = jax.random.split(key)
    state_noise = jax.random.multivariate_normal(key_state, jnp.zeros(state_size), Q, (nsteps,))
    obs_noise = jax.random.multivariate_normal(key_obs, jnp.zeros(obs_size), R, (nsteps,))
    simulation = _rk2(x0, fz, nsteps, dt)

    if noisy:
        simulation = simulation + jnp.sqrt(dt) * state_noise

    sample_state = simulation[::jump_size]
    sample_obs = jnp.apply_along_axis(fx, 1, sample_state) + obs_noise[:len(sample_state)]

    return sample_state, sample_obs, jump_size


def _Vt_dot(V, G, Q):
    return G @ V @ G.T + Q


def estimate(params: NLDS,
             sample_state: chex.Array,
             sample_obs: chex.Array,
             jump_size: int,
             dt: float):
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

    fz, fx = params.fz, params.fx
    Q, R = params.Qz, params.Rx

    Dfz = jacrev(fz)
    Dfx = jacrev(fx)

    state_size, _ = Q.shape
    obs_size, _ = R.shape

    I = jnp.eye(state_size)
    Vt = R.copy()
    mu_t = sample_state[0]

    def jump_step(state, t):
        mu_t, Vt = state
        k1 = fz(mu_t)
        k2 = fz(mu_t + dt * k1)
        mu_t = mu_t + dt * (k1 + k2) / 2

        Gt = Dfz(mu_t)
        k1 = _Vt_dot(Vt, Gt, Q)
        k2 = _Vt_dot(Vt + dt * k1, Gt, Q)
        Vt = Vt + dt * (k1 + k2) / 2
        return (mu_t, Vt), None

    def step(state, obs):
        jumps = jnp.arange(jump_size)
        (mu, V), _ = lax.scan(jump_step, state, jumps)

        mu_t_cond = mu
        Vt_cond = V
        Ht = Dfx(mu_t_cond)

        Kt = Vt_cond @ Ht.T @ jnp.linalg.inv(Ht @ Vt_cond @ Ht.T + R)
        mu = mu_t_cond + Kt @ (obs - fx(mu_t_cond))
        V = (I - Kt @ Ht) @ Vt_cond
        return (mu, V), (mu, V)

    initial_state = (mu_t.copy(), Vt.copy())
    _, (mu_hist, V_hist) = lax.scan(step, initial_state, sample_obs[1:])

    mu_hist = jnp.vstack([mu_t, mu_hist])
    V_hist = jnp.vstack([Vt, V_hist])

    return mu_hist, V_hist
