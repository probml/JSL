import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

class NonMarkovianSequenceModel:
    """
    Non-Markovian Gaussian Sequence Model
    """
    def __init__(self, phi, beta, q, r):
        """
        Parameters
        ----------
        phi: float
            Multiplicative effect in latent-space
        beta: float
            Decay relate in observed-space
        q: float
            Variance in latent-space
        r: float
            Variance in observed-space
        """
        self.phi = phi
        self.beta = beta
        self.q = q
        self.r = r
    
    @staticmethod
    def _obtain_weights(log_weights):
        weights = jnp.exp(log_weights - jax.nn.logsumexp(log_weights))
        return weights

    def sample_latent_step(self, key, x_prev):
        x_next = jax.random.normal(key) * jnp.sqrt(self.q) + self.phi * x_prev
        return x_next
    
    def sample_observed_step(self, key, mu, x_curr):
        mu_next = self.beta * mu + x_curr
        y_curr = jax.random.normal(key) * jnp.sqrt(self.r) + mu_next
        return y_curr, mu_next
    
    def sample_step(self, key, x_prev, mu_prev):
        key_latent, key_obs = jax.random.split(key)
        x_curr = self.sample_latent_step(key_latent, x_prev)
        y_curr, mu = self.sample_observed_step(key_obs, mu_prev, x_curr)
        
        carry_vals = {"x": x_curr, "y": y_curr}
        return (x_curr, mu), carry_vals
    
    def sample_single(self, key, nsteps):
        """
        Sample a single path from the non-Markovian Gaussian state-space model.
        
        Parameters
        ----------
        key: jax.random.PRNGKey
            Initial seed
        
        """
        key_init, key_simul = jax.random.split(key)
        x_init = jax.random.normal(key_init) * jnp.sqrt(self.q)
        mu_init = 0
        
        keys = jax.random.split(key_simul, nsteps)    
        carry_init = (x_init, mu_init)
        _, hist = jax.lax.scan(lambda carry, key: self.sample_step(key, *carry), carry_init, keys)
        return hist
    
    def sample(self, key, nsteps, nsims=1):
        """
        Sample from a non-Markovian Gaussian state-space model.
        
        Parameters
        ----------
        key: jax.random.PRNGKey
            Initial key to perform the simulation.
        nsteps: int
            Total number of steps to sample.
        nsims: int
            Number of paths to sample.
        """
        key_simulations = jax.random.split(key, nsims)
        sample_vmap = jax.vmap(self.sample_single, (0, None))
        
        simulations = sample_vmap(key_simulations, nsteps)
        
        # convert to one-dimensional array if only one simulation is
        # required
        if nsims == 1:
            for key, values in simulations.items():
                simulations[key] = values.ravel()
        
        return simulations
    
    def _sis_step(self, key, log_weights_prev, mu_prev, xparticles_prev, yobs):
        """
        Compute one step of the sequential-importance-sampling algorithm
        at time t.
        
        Parameters
        ----------
        key: jax.random.PRNGKey
            key to sample particle.
        mu_prev: array(n_particles)
            Term carrying past cumulate values.
        xsamp_prev: array(n_particles)
            Samples / particles from the latent space at t-1
        yobs: float
            Observation at time t.
        """
        
        key_particles = jax.random.split(key, len(xparticles_prev))
        
        # 1. Sample from proposal
        xparticles = jax.vmap(self.sample_latent_step)(key_particles, xparticles_prev)
        # 2. Evaluate unnormalised weights
        # 2.1 Compute new mean
        mu = self.beta * mu_prev + xparticles
        # 2.2 Compute log-unnormalised weights
        
        log_weights = log_weights_prev + norm.logpdf(yobs, loc=mu, scale=jnp.sqrt(self.q))
        
        return (log_weights, mu, xparticles), log_weights
    
    def sequential_importance_sample(self, key, observations, n_particles=10):
        """
        Apply sequential importance sampling to a series of observations. Sampling
        considers the transition distribution as the proposal.
        
        Parameters
        ----------
        key: jax.random.PRNGKey
            Initial key.
        observations: array(n_observations)
            one-array of observed values.
        n_particles: int (default: 10)
            Total number of particles to consider in the SIS filter.
        """
        T = len(observations)
        key, key_init_particles = jax.random.split(key)
        keys = jax.random.split(key, T)
        
        init_log_weights = jnp.zeros(n_particles)
        init_mu = jnp.zeros(n_particles) # equiv. ∀n.wn=1.0
        init_xparticles = jax.random.normal(key_init_particles, shape=(n_particles,)) * jnp.sqrt(self.q)
        
        carry_init = (init_log_weights, init_mu, init_xparticles)
        xs_tuple = (keys, observations)
        
        _, log_weights = jax.lax.scan(lambda carry, xs: self._sis_step(xs[0], *carry, xs[1]), carry_init, xs_tuple)
        return log_weights
    
    def _smc_step(self, key, log_weights_prev, mu_prev, xparticles_prev, yobs):
        n_particles = len(xparticles_prev)
        key, key_particles = jax.random.split(key)
        key_particles = jax.random.split(key_particles, n_particles)
        
        # 1. Resample particles
        weights = self._obtain_weights(log_weights_prev)
        ix_sampled = jax.random.choice(key, n_particles, p=weights, shape=(n_particles,))
        xparticles_prev_sampled = xparticles_prev[ix_sampled]
        mu_prev_sampled = mu_prev[ix_sampled]
        # 2. Propagate particles
        xparticles = jax.vmap(self.sample_latent_step)(key_particles, xparticles_prev_sampled)
        # 3. Concatenate
        mu = self.beta * mu_prev_sampled + xparticles

        # ToDo: return dictionary of log_weights and sampled indices
        log_weights = norm.logpdf(yobs, loc=mu, scale=jnp.sqrt(self.q))
        dict_carry = {
            "log_weights": log_weights,
            "indices": ix_sampled
        }
        return (log_weights, mu, xparticles_prev_sampled), dict_carry

    
    def sequential_monte_carlo(self, key, observations, n_particles=10):
        """
        Apply sequential Monte Carlo (SCM), a.k.a sequential importance resampling (SIR),
        a.k.a sequential importance sampling and resampling(SISR).
        """
        T = len(observations)
        key, key_particle_init = jax.random.split(key)
        keys = jax.random.split(key, T)
        
        init_xparticles = jax.random.normal(key_particle_init, shape=(n_particles,)) * jnp.sqrt(self.q)
        init_log_weights = jnp.zeros(n_particles) # equiv. ∀n.wn=1.0
        init_mu = jnp.zeros(n_particles)
        
        carry_init = (init_log_weights, init_mu, init_xparticles)
        xs_tuple = (keys, observations)
        _, dict_hist = jax.lax.scan(lambda carry, xs: self._smc_step(xs[0], *carry, xs[1]), carry_init, xs_tuple)
        # transform log-unnormalised weights to weights
        dict_hist["weights"] = jnp.exp(dict_hist["log_weights"] - jax.nn.logsumexp(dict_hist["log_weights"], axis=1, keepdims=True))
        
        return dict_hist
    