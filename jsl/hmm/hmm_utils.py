# Common functions that can be used for any hidden markov model type.
# Author: Aleyna Kara(@karalleyna)

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap, jit
from jax.random import split, randint, PRNGKey, permutation
from functools import partial
# !pip install graphviz
from graphviz import Digraph


@partial(jit, static_argnums=(2,))
def hmm_sample_minibatches(observations, valid_lens, batch_size, rng_key):
    '''
    Creates minibatches consists of the random permutations of the
    given observation sequences

    Parameters
    ----------
    observations : array(N, seq_len)
        All observation sequences

    valid_lens : array(N, seq_len)
        Consists of the valid length of each observation sequence

    batch_size : int
        The number of observation sequences that will be included in
        each minibatch

    rng_key : array
        Random key of shape (2,) and dtype uint32

    Returns
    -------
    * array(num_batches, batch_size, max_len)
        Minibatches
    '''
    num_train = len(observations)
    perm = permutation(rng_key, num_train)

    def create_mini_batch(batch_idx):
        return observations[batch_idx], valid_lens[batch_idx]

    num_batches = num_train // batch_size
    batch_indices = perm.reshape((num_batches, -1))
    minibatches = vmap(create_mini_batch)(batch_indices)
    return minibatches


@partial(jit, static_argnums=(1, 2, 3))
def hmm_sample_n(params, sample_fn, n, max_len, rng_key):
    '''
    Generates n observation sequences from the given Hidden Markov Model

    Parameters
    ----------
    params : HMMNumpy or HMMJax
        Hidden Markov Model

    sample_fn :
        The sample function of the given hidden markov model

    n : int
        The total number of observation sequences

    max_len : int
        The upper bound of the length of each observation sequence. Note that the valid length of the observation
        sequence is less than or equal to the upper bound.

    rng_key : array
        Random key of shape (2,) and dtype uint32

    Returns
    -------
    * array(n, max_len)
        Observation sequences
    '''

    def sample_(params, n_samples, key):
        return sample_fn(params, n_samples, key)[1]

    rng_key, rng_lens = split(rng_key)
    lens = randint(rng_lens, (n,), minval=1, maxval=max_len + 1)
    keys = split(rng_key, n)
    observations = vmap(sample_, in_axes=(None, None, 0))(params, max_len, keys)
    return observations, lens


@jit
def pad_sequences(observations, valid_lens, pad_val=0):
    '''
    Generates n observation sequences from the given Hidden Markov Model

    Parameters
    ----------

    observations : array(N, seq_len)
        All observation sequences

    valid_lens : array(N, seq_len)
        Consists of the valid length of each observation sequence

    pad_val : int
        Value that the invalid observable events of the observation sequence will be replaced

    Returns
    -------
    * array(n, max_len)
        Ragged dataset
    '''

    def pad(seq, len):
        idx = jnp.arange(1, seq.shape[0] + 1)
        return jnp.where(idx <= len, seq, pad_val)

    ragged_dataset = vmap(pad, in_axes=(0, 0))(observations, valid_lens), valid_lens
    return ragged_dataset


def hmm_plot_graphviz(trans_mat, obs_mat, init_dist, file_name, states=[], observations=[]):
    """
    Visualizes HMM transition matrix and observation matrix using graphhiz.

    Parameters
    ----------
    trans_mat, obs_mat, init_dist: arrays

    file_name : str
        Name of file which stores the output.
        The function creates file_name.pdf and file_name; the latter is a .dot text file.

    states: List(num_hidden)
        Names of hidden states

    observations: List(num_obs)
        Names of observable events

    Returns
    -------
    dot object, that can be displayed in colab
    """

    n_states, n_obs = obs_mat.shape

    dot = Digraph(comment='HMM')
    if not states:
        states = [f'State {i + 1}' for i in range(n_states)]
    if not observations:
        observations = [f'Obs {i + 1}' for i in range(n_obs)]

    # Creates hidden state nodes
    for i, name in enumerate(states):
        table = [f'<TR><TD>{observations[j]}</TD><TD>{"%.2f" % prob}</TD></TR>' for j, prob in
                 enumerate(obs_mat[i])]
        label = f'''<<TABLE><TR><TD BGCOLOR="lightblue" COLSPAN="2">{name}</TD></TR>{''.join(table)}</TABLE>>'''
        dot.node(f's{i}', label=label)

    # Writes transition probabilities
    for i in range(n_states):
        for j in range(n_states):
            dot.edge(f's{i}', f's{j}', label=str('%.2f' % trans_mat[i, j]))
    dot.attr(rankdir='LR')
    # dot.render(file_name, view=True)
    return dot
