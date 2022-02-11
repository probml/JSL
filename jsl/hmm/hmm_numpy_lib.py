# Inference and learning code for Hidden Markov Models using discrete observations.
# Has Numpy version of each function. For the Jax version, please see hmm_lib.py
# The Jax version of inference (not learning)
# has been upstreamed to https://github.com/deepmind/distrax/blob/master/distrax/_src/utils/hmm.py.
# This version is kept for historical purposes.
# Author: Gerardo Duran-Martin (@gerdm), Aleyna Kara (@karalleyna), Kevin Murphy (@murphyk)

from numpy.random import seed
import numpy as np
from scipy.special import softmax
from dataclasses import dataclass

import jax
from jax.nn import softmax

import superimport


'''
Hidden Markov Model class used in numpy implementations of inference algorithms.
'''
@dataclass
class HMMNumpy:
    trans_mat: np.array  # A : (n_states, n_states)
    obs_mat: np.array  # B : (n_states, n_obs)
    init_dist: np.array  # pi : (n_states)


def normalize_numpy(u, axis=0, eps=1e-15):
    '''
    Normalizes the values within the axis in a way that they sum up to 1.

    Parameters
    ----------
    u : array
    axis : int
    eps : float
        Threshold for the alpha values

    Returns
    -------
    * array
        Normalized version of the given matrix

    * array(seq_len, n_hidden) :
        The values of the normalizer
    '''
    u = np.where(u == 0, 0, np.where(u < eps, eps, u))
    c = u.sum(axis=axis)
    c = np.where(c == 0, 1, c)
    return u / c, c


def hmm_sample_numpy(params, seq_len, random_state=0):
    '''
    Samples an observation of given length according to the defined
    hidden markov model and gives the sequence of the hidden states
    as well as the observation.

    Parameters
    ----------
    params : HMMNumpy
        Hidden Markov Model

    seq_len: array(seq_len)
        The length of the observation sequence

    random_state : int
        Seed value

    Returns
    -------
    * array(seq_len,)
        Hidden state sequence

    * array(seq_len,) :
        Observation sequence
    '''

    def sample_one_step_(hist, a, p):
        x_t = np.random.choice(a=a, p=p)
        return np.append(hist, [x_t]), x_t

    seed(random_state)

    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape

    state_seq = np.array([], dtype=int)
    obs_seq = np.array([], dtype=int)

    latent_states = np.arange(n_states)
    obs_states = np.arange(n_obs)

    state_seq, zt = sample_one_step_(state_seq, latent_states, init_dist)
    obs_seq, xt = sample_one_step_(obs_seq, obs_states, obs_mat[zt])

    for _ in range(1, seq_len):
        state_seq, zt = sample_one_step_(state_seq, latent_states, trans_mat[zt])
        obs_seq, xt = sample_one_step_(obs_seq, obs_states, obs_mat[zt])

    return state_seq, obs_seq


##############################
# Inference

def hmm_forwards_numpy(params, obs_seq, length):
    '''
    Calculates a belief state

    Parameters
    ----------
    params : HMMNumpy
        Hidden Markov Model

    obs_seq: array(seq_len)
        History of observable events

    Returns
    -------
    * float
        The loglikelihood giving log(p(x|model))

    * array(seq_len, n_hidden) :
        All alpha values found for each sample
    '''
    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape
    seq_len = len(obs_seq)

    alpha_hist = np.zeros((seq_len, n_states))
    ll_hist = np.zeros(seq_len)  # loglikelihood history

    alpha_n = init_dist * obs_mat[:, obs_seq[0]]
    alpha_n, cn = normalize_numpy(alpha_n)

    alpha_hist[0] = alpha_n
    ll_hist[0] = np.log(cn)

    for t in range(1, length):
        alpha_n = obs_mat[:, obs_seq[t]] * (alpha_n[:, None] * trans_mat).sum(axis=0)
        alpha_n, cn = normalize_numpy(alpha_n)

        alpha_hist[t] = alpha_n
        ll_hist[t] = np.log(cn) + ll_hist[t - 1]  # calculates the loglikelihood up to time t

    return ll_hist[length - 1], alpha_hist


def hmm_loglikelihood_numpy(params, observations, lens):
    '''
    Finds the loglikelihood of each observation sequence sequentially.

    Parameters
    ----------
    params : HMMNumpy
        Hidden Markov Model

    observations: array(N, seq_len)
        Batch of observation sequences

    lens : array(N, seq_len)
        Consists of the valid length of each observation sequence

    Returns
    -------
    * array(N, seq_len)
        Consists of the loglikelihood of each observation sequence
    '''
    return np.array([hmm_forwards_numpy(params, obs, length)[0] for obs, length in zip(observations, lens)])


def hmm_backwards_numpy(params, obs_seq, length=None):
    '''
    Computes the backwards probabilities

    Parameters
    ----------
    params : HMMNumpy
        Hidden Markov Model

    obs_seq: array(seq_len,)
        History of observable events

    length : array(seq_len,)
        The valid length of the observation sequence

    Returns
    -------
    * array(seq_len, n_states)
       Beta values
    '''
    seq_len = len(obs_seq)

    if length is None:
        length = seq_len

    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist

    n_states, n_obs = obs_mat.shape
    beta_next = np.ones(n_states)

    beta_hist = np.zeros((seq_len, n_states))
    beta_hist[-1] = beta_next

    for t in range(2, length + 1):
        beta_next, _ = normalize_numpy((beta_next * obs_mat[:, obs_seq[-t + 1]] * trans_mat).sum(axis=1))
        beta_hist[-t] = beta_next

    return beta_hist


def hmm_forwards_backwards_numpy(params, obs_seq, length=None):
    '''
    Computes, for each time step, the marginal conditional probability that the Hidden Markov Model was
    in each possible state given the observations that were made at each time step, i.e.
    P(z[i] | x[0], ..., x[num_steps - 1]) for all i from 0 to num_steps - 1

    Parameters
    ----------
    params : HMMNumpy
        Hidden Markov Model

    obs_seq: array(seq_len)
        History of observed states

    Returns
    -------
    * array(seq_len, n_states)
        Alpha values

    * array(seq_len, n_states)
        Beta values

    * array(seq_len, n_states)
        Marginal conditional probability

    * float
        The loglikelihood giving log(p(x|model))
    '''
    seq_len = len(obs_seq)
    if length is None:
        length = seq_len

    ll, alpha = hmm_forwards_numpy(params, obs_seq, length)
    beta = hmm_backwards_numpy(params, obs_seq, length)

    gamma = alpha * np.roll(beta, -seq_len + length, axis=0)
    normalizer = gamma.sum(axis=1, keepdims=True)
    gamma = gamma / np.where(normalizer == 0, 1, normalizer)

    return alpha, beta, gamma, ll


def hmm_viterbi_numpy(params, obs_seq):
    """
    Compute the most probable sequence of states

    Parameters
    ----------
    params : HMMNumpy
        Hidden Markov Model

    obs_seq: array(seq_len)
        History of observed states

    Returns
    -------
    * array(seq_len)
        Sequence of most MAP probable sequence of states
    """
    seq_len = len(obs_seq)

    trans_mat, obs_mat, init_dist = np.log(params.trans_mat), np.log(params.obs_mat), np.log(params.init_dist)

    n_states, _ = obs_mat.shape

    first_prob = init_dist + obs_mat[:, obs_seq[0]]

    if len(obs_seq) == 1:
        return np.expand_dims(np.argmax(first_prob), axis=0)

    prev_prob = first_prob
    most_likely_sources = []

    for obs in obs_seq[1:]:
        obs_prob = obs_mat[..., obs]
        p = prev_prob[..., None] + trans_mat + obs_prob[..., None, :]
        max_p_given_successor = np.max(p, axis=-2)
        most_likely_given_successor = np.argmax(p, axis=-2)
        prev_prob = max_p_given_successor
        most_likely_sources.append(most_likely_given_successor)

    final_prob = prev_prob
    final_state = np.argmax(final_prob)

    most_likely_initial_given_successor = np.argmax(
        trans_mat + final_prob, axis=-2)

    most_likely_sources = np.vstack([
        np.expand_dims(most_likely_initial_given_successor, axis=0),
        np.array(most_likely_sources)])

    most_likely_path, state = [], final_state

    for most_likely_source in reversed(most_likely_sources[1:]):
        state = jax.nn.one_hot(state, n_states)
        most_likely = np.sum(most_likely_source * state).astype(np.int64)
        state = most_likely
        most_likely_path.append(most_likely)

    return np.append(np.flip(most_likely_path), final_state)


###############
# Learning using EM (Baum Welch)

@dataclass
class PriorsNumpy:
    trans_pseudo_counts: np.array
    obs_pseudo_counts: np.array
    init_pseudo_counts: np.array


def init_random_params_numpy(sizes, random_state):
    """
    Initializes the components of HMM from normal distibution

    Parameters
    ----------
    sizes: List
        Consists of the number of hidden states and observable events, respectively

    random_state : int
        Seed value

    Returns
    -------
    * HMMNumpy
        Hidden Markov Model
    """
    num_hidden, num_obs = sizes
    np.random.seed(random_state)
    return HMMNumpy(softmax(np.random.randn(num_hidden, num_hidden), axis=1),
                    softmax(np.random.randn(num_hidden, num_obs), axis=1),
                    softmax(np.random.randn(num_hidden)))


def compute_expected_trans_counts_numpy(params, alpha, beta, obs, T):
    """
    Computes the expected transition counts by summing ksi_{jk} for the observation given for all states j and k.
    ksi_{jk} for any time t in [0, T-1] can be calculated as the multiplication of the probability of ending
    in state j at t, the probability of starting in state k at t+1, the transition probability a_{jk} and b_{k obs[t+1]}.
    Note that ksi[t] is normalized so that the probabilities sums up to 1 for each time t in [0, T-1].

    Parameters
    ----------
    params: HMMNumpy
       Hidden Markov Model

    alpha : array
        A matrix of shape (seq_len, n_states)

    beta : array
        A matrix of shape (seq_len, n_states)

    obs : array
        One observation sequence

    T : int
        The valid length of observation sequence

    Returns
    ----------

    * array
        The matrix of shape (n_states, n_states) representing expected transition counts given obs o.
    """
    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape

    AA = np.zeros((n_states, n_states))  # AA[,j,k] = sum_t p(z(t)=j, z(t+1)=k|obs)

    for t in range(T - 1):
        ksi = alpha[t] * trans_mat.T * beta[t + 1] * obs_mat[:, obs[t + 1]]
        normalizer = ksi.sum()
        ksi /= 1 if normalizer == 0 else ksi.sum()
        AA += ksi.T
    return AA


def compute_expected_obs_counts_numpy(gamma, obs, T, n_states, n_obs):
    """
    Computes the expected observation count for each observation o by summing the probability of being at any of the
    states for each time t.
    Parameters
    ----------
    gamma : array
        A matrix of shape (seq_len, n_states)

    obs : array
        An array of shape (seq_len,)

    T : int
        The valid length of observation sequence

    n_states : int
        The number of hidden states

    n_obs : int
        The number of observable events

    Returns
    ----------
    * array
        A matrix of shape (n_states, n_obs) representing expected observation counts given observation sequence.
    """
    BB = np.zeros((n_states, n_obs))
    for t in range(T):
        o = obs[t]
        BB[:, o] += gamma[t]
    return BB


def hmm_e_step_numpy(params, observations, valid_lengths):
    """

    Calculates the the expectation of the complete loglikelihood over the distribution of
    observations given the current parameters

    Parameters
    ----------
    params: HMMNumpy
       Hidden Markov Model

    observations : array
        All observation sequences

    valid_lengths : array
        Valid lengths of each observation sequence

    Returns
    ----------
    * array
        A matrix of shape (n_states, n_states) representing expected transition counts

    * array
        A matrix of shape (n_states, n_obs) representing expected observation counts

    * array
        An array of shape (n_states,) representing expected initial counts calculated from summing gamma[0] of each
        observation sequence

    * float
        The sum of the likelihood, p(o | lambda) where lambda stands for (trans_mat, obs_mat, init_dist) triple, for
        each observation sequence o.
    """
    N, _ = observations.shape

    trans_mat, obs_mat, init_dist = params.trans_mat, params.obs_mat, params.init_dist
    n_states, n_obs = obs_mat.shape

    trans_counts = np.zeros((n_states, n_states))
    obs_counts = np.zeros((n_states, n_obs))
    init_counts = np.zeros((n_states))

    loglikelihood = 0

    for obs, valid_len in zip(observations, valid_lengths):
        alpha, beta, gamma, ll = hmm_forwards_backwards_numpy(params, obs, valid_len)
        trans_counts = trans_counts + compute_expected_trans_counts_numpy(params, alpha, beta, obs, valid_len)
        obs_counts = obs_counts + compute_expected_obs_counts_numpy(gamma, obs, valid_len, n_states, n_obs)
        init_counts = init_counts + gamma[0]
        loglikelihood += ll

    return trans_counts, obs_counts, init_counts, loglikelihood


def hmm_m_step_numpy(counts, priors=None):
    """

    Recomputes new parameters from A, B and pi using max likelihood.

    Parameters
    ----------
    counts: tuple
        Consists of expected transition counts, expected observation counts, and expected initial state counts,
        respectively.

    priors : PriorsNumpy

    Returns
    ----------
    * HMMNumpy
        Hidden Markov Model

    """
    trans_counts, obs_counts, init_counts = counts

    if priors is not None:
        trans_counts = trans_counts + priors.trans_pseudo_counts
        obs_counts = obs_counts + priors.obs_pseudo_count
        init_counts = init_counts + priors.init_pseudo_counts

    A = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    B = obs_counts / obs_counts.sum(axis=1, keepdims=True)
    pi = init_counts / init_counts.sum()

    return HMMNumpy(A, B, pi)


def hmm_em_numpy(observations, valid_lengths, n_hidden=None, n_obs=None,
                 init_params=None, priors=None, num_epochs=1, random_state=None):
    """
    Implements Baum–Welch algorithm which is used for finding its components, A, B and pi.

    Parameters
    ----------
    observations: array
        All observation sequences

    valid_lengths : array
        Valid lengths of each observation sequence

    n_hidden : int
        The number of hidden states

    n_obs : int
        The number of observable events

    init_params : HMMNumpy
        Initial Hidden Markov Model

    priors : PriorsNumpy
        Priors for the components of Hidden Markov Model

    num_epochs : int
        Number of times model will be trained

    random_state: int
        Seed value

    Returns
    ----------
    * HMMNumpy
        Trained Hidden Markov Model

    * array
        Negative loglikelihoods each of which can be interpreted as the loss value at the current iteration.
    """

    if random_state is None:
        random_state = 0

    if init_params is None:
        try:
            init_params = init_random_params_numpy([n_hidden, n_obs], random_state)
        except:
            raise ValueError("n_hidden and n_obs should be specified when init_params was not given.")

    neg_loglikelihoods = []
    params = init_params

    for _ in range(num_epochs):
        trans_counts, obs_counts, init_counts, ll = hmm_e_step_numpy(params, observations, valid_lengths)
        neg_loglikelihoods.append(-ll)
        params = hmm_m_step_numpy([trans_counts, obs_counts, init_counts], priors)

    return params, neg_loglikelihoods
