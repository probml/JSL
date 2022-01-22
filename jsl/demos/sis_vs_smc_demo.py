# This demo compares sequential importance sampling (SIS) to
# sequential Monte Carlo (SMC) in the case of a non-markovian 
# Gaussian sequence model.

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jsl.nlds.sequential_monte_carlo import NonMarkovianSequenceModel

def find_path(ix_path, final_state):
    curr_state = final_state
    path = [curr_state]
    for i in range(1, 7):
        curr_state, _ = ix_path[:, -i, curr_state]
        path.append(curr_state)
    path = path[::-1]
    return path


def plot_sis_weights(hist, n_steps, spacing=1.5, max_size=0.3):
    """
    Plot the evolution of weights in the sequential importance sampling (SIS) algorithm.

    Parameters
    ----------
    weights: array(n_particles, n_steps)
        Weights at each time step.
    n_steps: int
        Number of steps to plot.
    spacing: float
        Spacing between particles.
    max_size: float
        Maximum size of the particles.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect(1)
    weights_subset = hist["weights"][:n_steps]
    for col, weights_row in enumerate(weights_subset):
        norm_cst = weights_row.sum()
        radii = weights_row / norm_cst * max_size
        for row, rad in enumerate(radii):
            if col != n_steps - 1:
                plt.arrow(spacing * (col + 0.25), row, 0.6, 0, width=0.05,
                          edgecolor="white", facecolor="tab:gray")
            circle = plt.Circle((spacing * col, row), rad, color="tab:red")
            ax.add_artist(circle)

    plt.xlim(-1, n_steps * spacing)
    plt.xlabel("Iteration (t)")
    plt.ylabel("Particle index (i)")

    xticks_pos = jnp.arange(0, n_steps * spacing - 1, 2)
    xticks_lab = jnp.arange(1, n_steps + 1)
    plt.xticks(xticks_pos, xticks_lab)

    return fig, ax


def plot_smc_weights(hist, n_steps, spacing=1.5, max_size=0.3):
    """
    Plot the evolution of weights in the sequential Monte Carlo (SMC) algorithm.

    Parameters
    ----------
    weights: array(n_particles, n_steps)
        Weights at each time step.
    n_steps: int
        Number of steps to plot.
    spacing: float
        Spacing between particles.
    max_size: float
        Maximum size of the particles.
    
    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect(1)

    weights_subset = hist["weights"][:n_steps]
    # sampled indices represent the "position" of weights at the next time step
    ix_subset = hist["indices"][:n_steps][1:]

    for it, (weights_row, p_target) in enumerate(zip(weights_subset, ix_subset)):
        norm_cst = weights_row.sum()
        radii = weights_row / norm_cst * max_size
        
        for particle_ix, (rad, target_ix) in enumerate(zip(radii, p_target)):
            if it != n_steps - 2:
                diff = particle_ix - target_ix
                plt.arrow(spacing * (it + 0.15), target_ix, 1.3, diff, width=0.05,
                        edgecolor="white", facecolor="tab:gray", length_includes_head=True)
            circle = plt.Circle((spacing * it, particle_ix), rad, color="tab:blue")
            ax.add_artist(circle)

    plt.xlim(-1, n_steps * spacing - 2)
    plt.xlabel("Iteration (t)")
    plt.ylabel("Particle index (i)")

    xticks_pos = jnp.arange(0, n_steps * spacing - 2, 2)
    xticks_lab = jnp.arange(1, n_steps)
    plt.xticks(xticks_pos, xticks_lab)

    # ylims = ax.axes.get_ylim() # to-do: grab this value for SCM-particle descendents' plot

    return fig, ax


def plot_smc_weights_unique(hist, n_steps, spacing=1.5, max_size=0.3):
    """
    We plot the evolution of particles that have been consistently resampled and form our final
    approximation of the target distribution using sequential Monte Carlo (SMC).

    Parameters
    ----------
    weights: array(n_particles, n_steps)
        Weights at each time step.
    n_steps: int
        Number of steps to plot.
    spacing: float
        Spacing between particles.
    max_size: float
        Maximum size of the particles.
    

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure containing the plot.
    """
    weights_subset = hist["weights"][:n_steps]
    # sampled indices represent the "position" of weights at the next time step
    ix_subset = hist["indices"][:n_steps][1:]
    ix_path = ix_subset[:n_steps - 2]
    ix_map = jnp.repeat(jnp.arange(5)[None, :], 6, axis=0)
    ix_path = jnp.stack([ix_path, ix_map], axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect(1)

    for final_state in range(5):
        path = find_path(ix_path, final_state)
        path_beg, path_end = path[:-1], path[1:]
        for it, (beg, end) in enumerate(zip(path_beg, path_end)):
            diff = end - beg
            plt.arrow(spacing * (it + 0.15), beg, 1.3, diff, width=0.05,
                    edgecolor="white", facecolor="tab:gray", alpha=1.0, length_includes_head=True)

    for it, weights_row in enumerate(weights_subset[:-1]):
        norm_cst = weights_row.sum()
        radii = weights_row / norm_cst * max_size
        
        for particle_ix, rad in enumerate(radii):
            circle = plt.Circle((spacing * it, particle_ix), rad, color="tab:blue")
            ax.add_artist(circle)
            
    plt.xlim(-1, n_steps * spacing - 2)
    plt.xlabel("Iteration (t)")
    plt.ylabel("Particle index (i)")

    xticks_pos = jnp.arange(0, n_steps * spacing - 2, 2)
    xticks_lab = jnp.arange(1, n_steps)

    plt.xticks(xticks_pos, xticks_lab)

    return fig, ax


def main():
    params = {
        "phi": 0.9,
        "q": 1.0,
        "beta": 0.5,
        "r": 1.0,
    }

    key = jax.random.PRNGKey(314)
    key_sample, key_sis, key_scm = jax.random.split(key, 3)
    seq_model = NonMarkovianSequenceModel(**params)
    hist_target = seq_model.sample(key_sample, 100)
    observations = hist_target["y"]

    res_sis = seq_model.sequential_importance_sample(key_sis, observations, n_particles=5)
    res_smc = seq_model.sequential_monte_carlo(key_scm, observations, n_particles=5)

    # Plot SMC particle evolution
    n_steps = 6 + 2
    spacing = 2

    dict_figures = {}

    fig, ax = plot_sis_weights(res_sis, n_steps=7, spacing=spacing)
    plt.tight_layout()
    dict_figures["sis_weights"] = fig

    fig, ax = plot_smc_weights(res_smc, n_steps=n_steps, spacing=spacing)
    ylims = ax.axes.get_ylim()
    plt.tight_layout()
    dict_figures["smc_weights"] = fig

    fig, ax = plot_smc_weights_unique(res_smc, n_steps=n_steps, spacing=spacing)
    ax.set_ylim(*ylims)
    plt.tight_layout()
    dict_figures["smc_weights_unique"] = fig

    return dict_figures


if __name__ == "__main__":
    from jsl.demos.plot_utils import savefig
    figures = main()
    savefig(figures)
    plt.show()