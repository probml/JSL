'''
As stated in the original paper Task Agnostic Continual Learning Using Online Variational Bayes with Fixed-Point Updates,

Foo-vb is the novel fixed-point equations for the online variational Bayes optimization problem,
 for multivariate Gaussian parametric distributions.

The original FOO-VB Pytorch implementation is available at https://github.com/chenzeno/FOO-VB.
This library is Jax implementation based on the original code.

Author: Aleyna Kara(@karalleyna)

'''
from jax.config import config
config.update("jax_enable_x64", True)

from jax.config import config
config.update("jax_debug_nans", True)

from random import randint
from jax import random

import flax.linen as nn
from typing import Sequence, Callable

import foo_vb_lib
import datasets as ds
import run

import ml_collections

import pickle

import numpy as np

from time import time

from jax import random, value_and_grad, tree_map, vmap, lax
import jax.numpy as jnp

from functools import partial

import foo_vb_lib

def error_fn(train_loader, test_loader, epochs):

    for task in range(len(test_loader)):
        for epoch in range(1, epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader[0]):
                continue

            for data, target in test_loader[task]:
                continue

        for i in range(task + 1):
            for data, target in test_loader[i]:
                continue


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.batch_size = 128
    config.test_batch_size = 1000

    config.epochs = 20
    config.seed = 1
    config.train_mc_iters = 3

    config.s_init = 0.27
    config.eta = 1.
    config.alpha = 0.5

    config.tasks = 10
    config.results_dir = "."

    config.dataset = "continuous_permuted_mnist"
    config.iterations_per_virtual_epc = 468

    config.diagonal = True

    return config


if __name__ == '__main__':
    config = get_config()
    config.alpha = 0.6
    
    key = random.PRNGKey(0)
    perm_key, key = random.split(key)

    image_size = 784
    n_permutations = 10

    permutations = foo_vb_lib.create_random_perm(perm_key, image_size, n_permutations)
    permutations = permutations[1:11]
    train_loaders, test_loaders = ds.ds_padded_cont_permuted_mnist(num_epochs=int(config.epochs * config.tasks),
                                                                   iterations_per_virtual_epc=config.iterations_per_virtual_epc,
                                                                   contpermuted_beta=4, permutations=permutations,
                                                                   batch_size=config.batch_size)

    error_fn(train_loaders, test_loaders, config.epochs)