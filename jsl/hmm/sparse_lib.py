"""
jax.experimental.sparse-compatible Hidden Markov Model (HMM)
"""
import jax
from functools import partial

def alpha_step(alpha_prev, y, local_evidence_multiple, transition_matrix):
    local_evidence = local_evidence_multiple(y)
    alpha_next = local_evidence * (transition_matrix.T @ alpha_prev)
    normalisation_cst = alpha_next.sum()
    alpha_next = alpha_next / normalisation_cst
    
    carry = {
        "alpha": alpha_next,
        "cst": normalisation_cst
    }
    
    return alpha_next, carry


def alpha_forward(obs, local_evidence, transition_matrix, alpha_init):
    alpha_step_part = partial(alpha_step,
                              local_evidence_multiple=local_evidence,
                              transition_matrix=transition_matrix)
    alpha_last, alpha_hist = jax.lax.scan(alpha_step_part, alpha_init, obs)
    return alpha_last, alpha_hist


def beta_step(beta_next, y, local_evidence_multiple, transition_matrix):
    norm_cst = beta_next.sum()
    local_evidence = local_evidence_multiple(y)
    beta_prev = transition_matrix @ (local_evidence * beta_next)
    beta_prev = beta_prev / norm_cst
        
    carry = {
        "beta": beta_prev,
        "cst": norm_cst
    }
    
    return beta_prev, carry


def beta_backward(obs, local_evidence, transition_matrix, alpha_last):
    beta_step_part = partial(beta_step,
                              local_evidence_multiple=local_evidence,
                              transition_matrix=transition_matrix)
    beta_first, beta_hist = jax.lax.scan(beta_step_part, alpha_last, obs, reverse=True)
    return beta_first, beta_hist
