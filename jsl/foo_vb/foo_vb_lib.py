'''
As stated in the original paper Task Agnostic Continual Learning Using Online Variational Bayes with Fixed-Point Updates,

Foo-vb is the novel fixed-point equations for the online variational Bayes optimization problem,
 for multivariate Gaussian parametric distributions.

The original FOO-VB Pytorch implementation is available at https://github.com/chenzeno/FOO-VB.
This library is Jax implementation based on the original code.

Author: Aleyna Kara(@karalleyna)

'''

import jax
import jax.numpy as jnp
from jax import random, vmap, tree_map, jit

from flax.core.frozen_dict import unfreeze, freeze
from flax import traverse_util

from functools import partial


@jit
def gen_phi(key, weights):
    phi = {}

    for k, v in weights.items():
        normal_key, key = random.split(key)
        phi[k] = random.normal(normal_key, shape=v.shape, dtype=jnp.float32)

    return phi


@jit
def update_weight(weights):
    """
        This function update the parameters of the network.
        :param weights: A list of matrices in size of P*N.
        :return:
    """
    params = {}
    for i, (k, v) in enumerate(weights.items()):
        params[k] = weights[k][:, :-1]
        params[tuple([*k[:-1], 'bias'])] = weights[k][:, -1]

    params = freeze(traverse_util.unflatten_dict(params))

    return params


@jit
def randomize_weights(m, a, b, phi):
    """
        This function generate a sample of normal random weights with mean M and covariance matrix of (A*A^t)\otimes(B*B^t)
        (\otimes = kronecker product). In matrix form the update rule is W = M + B*Phi*A^t.
        :param m: A list of matrices in size of P*N.
        :param a: A list of matrices in size of N*N.
        :param b: A list of matrices in size of P*P.
        :param phi: A list of normal random matrices in size of P*N.
        :return:
    """

    # W = M + B*Phi*A^t
    weights = jax.tree_multimap(lambda m, b, phi, a: m + ((b @ phi) @ a.T),
                                m, b, phi, a)
    params = update_weight(weights)
    return params


def cross_entropy_loss(params, inputs, labels, num_classes, predict_fn):
    logits = predict_fn(params, inputs)
    one_hot_labels = jax.nn.one_hot(labels, num_classes=num_classes)
    xentropy = (logits * one_hot_labels).sum(axis=-1)
    return -xentropy.sum()


@jit
def weight_grad(grads):
    """
        This function return a list of matrices containing the gradient of the network parameters for each layer.
        :return: grads: A list of matrices containing the gradients of the network parameters.
    """
    grad_mat = {}
    grads = unfreeze(grads)
    for k, v in traverse_util.flatten_dict(grads).items():
        if k[-1] == 'kernel':
            grad_mat[k] = jnp.hstack([v.T, grad_mat[k]])
        else:
            k = (*k[:-1], "kernel")
            grad_mat[k] = v[..., None]
    grads = freeze(grads)
    return grad_mat


@partial(jit, static_argnums=(1, 2))
def create_random_perm(key, image_size, n_permutations):
    """
        This function returns a list of array permutation (size of 28*28 = 784) to create permuted MNIST data.
        Note the first permutation is the identity permutation.
        :param n_permutations: number of permutations.
        :return permutations: a list of permutations.
    """
    initial_array = jnp.arange(image_size)
    keys = random.split(key, n_permutations - 1)

    def permute(key):
        return random.permutation(key, initial_array)

    permutation_list = vmap(permute)(keys)
    return jnp.vstack([initial_array, permutation_list])


@jit
def aggregate_grads(avg_psi, grads, train_mc_iters):
    """
        This function estimate the expectation of the gradient using Monte Carlo average.
        :param args: Training settings.
        :param avg_psi: A list of matrices in size of P*N.
        :param grads: A list of matrices in size of P*N.
        :return:
    """

    for k, v in grads.items():
        avg_psi[k] += (1 / train_mc_iters) * v

    return avg_psi


@jit
def aggregate_e_a(e_a, grads, b, phi, train_mc_iters):
    """
        This function estimate the expectation of the e_a ((1/P)E(Psi^t*B*Phi)) using Monte Carlo average.
        :param args: Training settings.
        :param e_a: A list of matrices in size of N*N.
        :param grads: A list of matrices in size of P*N.
        :param b: A list of matrices in size of P*P.
        :param phi: A list of normal random matrices in size of P*N.
        :return:
    """

    for k, v in grads.items():
        b, phi = b[k], phi[k]
        e_a[k] += (1 / (train_mc_iters * b.shape[0])) * ((v.T @ b) @ phi)
    return e_a


@jit
def aggregate_e_b(e_b, grads, a, phi, train_mc_iters):
    """
        This function estimate the expectation of the e_b ((1/N)E(Phi^t*A*Psi)) using Monte Carlo average.
        :param args: Training settings.
        :param e_b: A list of matrices in size of P*P.
        :param grads: A list of matrices in size of P*N.
        :param a: A list of matrices in size of N*N.
        :param phi: A list of normal random matrices in size of P*N
        :return:
    """
    for k, v in grads.items():
        a, phi = a[k], phi[k]
        e_b[k] += (1 / (train_mc_iters * a.shape[0])) * ((v @ a) @ phi.T)
    return e_b


def update_m(m, a, b, avg_psi, eta=1., diagonal=False):
    """
        This function updates the mean according to M = M - B*B^t*E[Psi]*A*A^t.
        :param m: m: A list of matrices in size of P*N.
        :param a: A list of matrices in size of N*N.
        :param b: A list of matrices in size of P*P.
        :param avg_psi: A list of matrices in size of P*N.
        :param eta: .
        :param diagonal: .
        :return:
    """

    if diagonal:
        # M = M - diag(B*B^t)*E[Psi]*diag(A*A^t)
        m = jax.tree_multimap(lambda m, b, psi, a: m - eta *
                                                   ((jnp.diag(jnp.diag(b @ b.T)) @ psi) @ jnp.diag(jnp.diag(a @ a.T)
                                                                                                   )),
                              m, b, avg_psi, a)
    else:
        # M = M - B*B^t*E[Psi]*A*A^t
        m = jax.tree_multimap(lambda m, b, psi, a: m - eta * ((b @ b.T) @ psi) @ (a @ a.T),
                              m, b, avg_psi, a)

    return m


@jit
def solve_matrix_equation(v_mat, e_mat):
    """
        This function returns a solution for the following non-linear matrix equation XX^{\top}+VEX^{\top}-V = 0.
        All the calculations are done in double precision.
        :param v_mat: N*N PD matrix.
        :param e_mat: N*N matrix.
        :param print_norm_flag: Boolean parameter. Print the norm of the matrix equation.
        :return: x_mat: N*N matrix a solution to the non-linear matrix equation.
    """

    v_mat = v_mat  # .astype(jnp.float64)
    e_mat = e_mat  # .astype(jnp.float64)

    ve_product = v_mat @ e_mat

    # B = V + (1/4)V*E*(E^T)*V
    b_mat = v_mat + 0.25 * (ve_product @ ve_product.T)

    # We don't need the full composition of b matrix. 
    left_mat, diag_mat, right_mat = jnp.linalg.svd(b_mat, full_matrices=False)

    '''
    The values of vs, which are returned by torch.svd and  jnp.linalg.svd are transpose of each other. So, we
    take the transpose of the right matrix in contrast to the original code.
    '''
    right_mat = right_mat.T

    # assert (jnp.min(diag_mat) > 0), "v_mat is singular!"

    # L = B^{1/2}
    l_mat = ((left_mat @ jnp.diag(jnp.sqrt(diag_mat))) @
             right_mat.T)

    inv_l_mat = (right_mat @ jnp.diag(1 / jnp.sqrt(diag_mat))) @ left_mat.T

    # L^-1*V*E=S*Lambda*W^t 
    # We don't need the full composition.
    s_mat, lambda_mat, w_mat = jnp.linalg.svd(inv_l_mat @ ve_product, full_matrices=False)

    '''
    The values of vs, which are returned by torch.svd and  jnp.linalg.svd are transpose of each other. So, we
    take the transpose of the w matrix in contrast to the original code.
    '''
    w_mat = w_mat.T

    # Q = S*W^t
    q_mat = s_mat @ w_mat.T

    # X = L*Q-(1/2)V*E
    x_mat = (l_mat @ q_mat) - 0.5 * ve_product
    return x_mat  # .astype(jnp.float64)


def update_a_b(a, b, e_a, e_b):
    """
        This function updates the matrices A & B using a solution to the non-linear matrix equation
        XX^{\top}+VEX^{\top}-V = 0.
        :param a:
        :param b:
        :param e_a:
        :param e_b:
        :return:
    """
    updated_a, updated_b = {}, {}
    for k, a in a.items():
        b = b[k]
        e_a = e_a[k]
        e_b = e_b[k]

        updated_a = solve_matrix_equation(a @ a.T, e_a)
        updated_b = solve_matrix_equation(b @ b.T, e_b)

        updated_a[k] = (updated_a)
        updated_b[k] = (updated_b)

    return updated_a, updated_b


@jit
def zero_matrix(avg_psi, e_a, e_b):
    """
        :param avg_psi: A list of matrices in size of P*N.
        :param e_a: A list of matrices in size of N*N.
        :param e_b: A list of matrices in size of P*P.
        :return:
    """
    avg_psi = jax.tree_map(lambda x: jnp.zeros(x.shape), avg_psi)
    e_a = jax.tree_map(lambda x: jnp.zeros(x.shape), e_a)
    e_b = jax.tree_map(lambda x: jnp.zeros(x.shape), e_b)
    return avg_psi, e_a, e_b


@partial(jit, static_argnums=(2, 3, 4))
def init_param(key, params, s_init, use_custom_init=False, alpha=0.5):
    """
        :param params: A list of iterators of the network parameters.
        :param s_init: Init value of the diagonal of a and b.
        :return: weights: A list of matrices in size of P*N.
        :return: m: A list of matrices in size of P*N.
        :return: a: A list of matrices in size of N*N.
        :return: b: A list of matrices in size of P*P.
        :return: avg_psi: A list of matrices in size of P*N.
        :return: e_a: A list of matrices in size of N*N.
        :return: e_b: A list of matrices in size of P*P.
    """

    weights = {}
    m = {}
    a = {}
    b = {}
    avg_psi = {}
    e_a = {}
    e_b = {}

    # Unfreeze params to normal dict.
    params = unfreeze(params)

    for k, v in traverse_util.flatten_dict(params).items():
        if k[-1] == 'kernel':
            out_feature, in_feature = v.shape

            w_mat = jnp.zeros((out_feature, in_feature + 1))
            weights[k] = w_mat

            avg_psi_mat = jnp.zeros((out_feature, in_feature + 1))
            avg_psi[k] = avg_psi_mat

            if use_custom_init:
                m_key, key = random.split(key)
                m_mat = jnp.sqrt((2.0 * alpha / (in_feature + 2.0))) * random.normal(m_key, shape=(
                    out_feature, in_feature + 1), dtype=jnp.float32)

                coef = jnp.sqrt(jnp.sqrt((2.0 * (1.0 - alpha) / (in_feature + 2.0))))
                a_mat = jnp.diag(coef * jnp.ones((in_feature + 1,)))
                b_mat = jnp.diag(coef * jnp.ones((out_feature,)))
            else:
                key1, key2, key = random.split(key, 3)
                m_mat = jnp.hstack([jnp.sqrt(2.0 / (out_feature + in_feature)) *
                                    random.normal(key1, shape=(out_feature, in_feature), dtype=jnp.float32),
                                    jnp.sqrt(2.0 / (1.0 + in_feature)) *
                                    random.normal(key2, shape=(out_feature, 1), dtype=jnp.float32)])

                a_mat = jnp.diag(s_init * jnp.ones((in_feature + 1,)))
                b_mat = jnp.diag(s_init * jnp.ones((out_feature,)))

            e_a_mat = jnp.zeros((v.shape[1] + 1, v.shape[1] + 1))
            e_b_mat = jnp.zeros((v.shape[0], v.shape[0]))

            m[k] = m_mat
            a[k] = a_mat
            b[k] = b_mat
            e_a[k] = e_a_mat
            e_b[k] = e_b_mat

    params = freeze(params)
    return weights, m, a, b, avg_psi, e_a, e_b
