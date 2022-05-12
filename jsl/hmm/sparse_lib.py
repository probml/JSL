"""
jax.experimental.sparse-compatible Hidden Markov Model (HMM)
"""

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
