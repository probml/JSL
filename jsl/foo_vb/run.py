'''

This script, functions of which are in foo_vb_lib.py, is based on
    https://github.com/chenzeno/FOO-VB/blob/ebc14a930ba9d1c1dadc8e835f746c567c253946/main.py
For more information, please see the original paper https://arxiv.org/abs/2010.00373 .
Author: Aleyna Kara(@karalleyna)
'''
import numpy as np

from time import time

from jax import random, value_and_grad, tree_map, vmap, lax
import jax.numpy as jnp

from functools import partial

import foo_vb_lib


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)


def init_step(key, model, image_size, config):
    model_key, param_key = random.split(key)
    variables = model.init(model_key, jnp.zeros((config.batch_size, image_size)))
    params = tree_map(jnp.transpose, variables)
    pytrees = foo_vb_lib.init_param(param_key, params, config.s_init, True, config.alpha)
    return pytrees


def train_step(key, pytrees, data, target, value_and_grad_fn, train_mc_iters, eta, diagonal):
    weights, m, a, b, avg_psi, e_a, e_b = pytrees

    def monte_carlo_step(aggregated_params, key):
        # Phi ~ MN(0,I,I)
        avg_psi, e_a, e_b = aggregated_params
        phi_key, key = random.split(key)
        phi = foo_vb_lib.gen_phi(phi_key, weights)

        # W = M +B*Phi*A^t
        params = foo_vb_lib.randomize_weights(m, a, b, phi)
        loss, grads = value_and_grad_fn(tree_map(jnp.transpose, params), data, target)
        grads = foo_vb_lib.weight_grad(grads)
        avg_psi = foo_vb_lib.aggregate_grads(avg_psi, grads, train_mc_iters)
        e_a = foo_vb_lib.aggregate_e_a(e_a, grads, b,
                                       phi, train_mc_iters)

        e_b = foo_vb_lib.aggregate_e_b(e_b, grads, a,
                                       phi, train_mc_iters)

        return (avg_psi, e_a, e_b), loss

    # M = M - B*B^t*avg_Phi*A*A^t
    keys = random.split(key, train_mc_iters)
    (avg_psi, e_a, e_b), losses = scan(monte_carlo_step,
                                       (avg_psi, e_a, e_b), keys)

    print("Loss :", losses.mean())

    m = foo_vb_lib.update_m(m, a, b, avg_psi, eta, diagonal=diagonal)
    a, b = foo_vb_lib.update_a_b(a, b, e_a, e_b)
    avg_psi, e_a, e_b = foo_vb_lib.zero_matrix(avg_psi, e_a, e_b)

    pytrees = weights, m, a, b, avg_psi, e_a, e_b

    return pytrees, losses


def eval_step(model, pytrees, data, target, train_mc_iters):
    weights, m, a, b, avg_psi, e_a, e_b = pytrees

    def monte_carlo_step(weights, phi_key):
        phi = foo_vb_lib.gen_phi(phi_key, weights)
        params = foo_vb_lib.randomize_weights(m, a, b, phi)
        output = model.apply(tree_map(jnp.transpose, params), data)
        # get the index of the max log-probability
        pred = jnp.argmax(output, axis=1)
        return weights, jnp.sum(pred == target)

    keys = random.split(random.PRNGKey(0), train_mc_iters)
    _, correct_per_iter = scan(monte_carlo_step, weights, keys)
    n_correct = jnp.sum(correct_per_iter)

    return n_correct


def train_continuous_mnist(key, model, train_loader,
                           test_loader, image_size, num_classes, config):
    init_key, key = random.split(key)
    pytrees = init_step(key, model, image_size, config)
    criterion = partial(foo_vb_lib.cross_entropy_loss,
                        num_classes=num_classes,
                        predict_fn=model.apply)

    grad_fn = value_and_grad(criterion)

    ava_test = []

    for task in range(len(test_loader)):
        for epoch in range(1, config.epochs + 1):
            start_time = time()
            for batch_idx, (data, target) in enumerate(train_loader[0]):
                data, target = jnp.array(data.view(-1, image_size).numpy()), jnp.array(target.numpy())

                train_key, key = random.split(key)
                pytrees, losses = train_step(train_key, pytrees, data, target, grad_fn,
                                             config.train_mc_iters, config.eta, config.diagonal)

            print("Time : ", time() - start_time)
            total = 0

            for data, target in test_loader[task]:
                data, target = jnp.array(data.numpy().reshape((-1, image_size))), jnp.array(target.numpy())
                n_correct = eval_step(model, pytrees, data, target, config.train_mc_iters)
                total += n_correct

            test_acc = 100. * total / (len(test_loader[task].dataset) * config.train_mc_iters)
            print('\nTask num {}, Epoch num {} Test Accuracy: {:.2f}%\n'.format(
                task, epoch, test_acc))

        test_accuracies = []

        for i in range(task + 1):
            total = 0
            for data, target in test_loader[i]:
                data, target = jnp.array(data.numpy().reshape((-1, image_size))), jnp.array(target.numpy())
                n_correct = eval_step(model, pytrees, data, target, config.train_mc_iters)
                total += n_correct

            test_acc = 100. * total / (len(test_loader[task].dataset) * config.train_mc_iters)
            test_accuracies.append(test_acc)

            print('\nTraning task Num: {} Test Accuracy of task {}: {:.2f}%\n'.format(
                task, i, test_acc))
        ava_test.append(jnp.mean(test_accuracies))

    return ava_test


def train_multiple_tasks(key, model, train_loader,
                         test_loader, num_classes,
                         permutations, image_size, config):
    init_key, key = random.split(key)
    pytrees = init_step(key, model, config)
    criterion = partial(foo_vb_lib.cross_entropy_loss,
                        num_classes=num_classes, predict_fn=model.apply)

    grad_fn = value_and_grad(criterion)

    ava_test = []

    for task in range(len(permutations)):
        for epoch in range(1, config.epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = jnp.array(data.detach().numpy().reshape((-1, image_size))), jnp.array(
                    target.detach().numpy())
                data = data[:, permutations[task]]

                train_key, key = random.split(key)
                start_time = time.time()
                pytrees, losses = train_step(train_key, pytrees, data, target, grad_fn,
                                             config.train_mc_iters, config.eta, config.diagonal)
                print("Time : ", start_time - time.time())

            total = 0

            for data, target in train_loader:
                data, target = jnp.array(data.numpy().reshape((-1, image_size))), jnp.array(target.numpy())
                data = data[:, permutations[task]]
                n_correct = eval_step(model, pytrees, data, target, config.train_mc_iters)
                total += n_correct

            train_acc = 100. * total / (len(train_loader.dataset) * config.train_mc_iters)

            total = 0

            for data, target in test_loader:
                data, target = jnp.array(data.numpy().reshape((-1, image_size))), jnp.array(target.numpy())
                data = data[:, permutations[task]]
                n_correct = eval_step(model, pytrees, data, target, config.train_mc_iters)
                total += n_correct

            test_acc = 100. * total / (len(test_loader.dataset) * config.train_mc_iters)
            print('\nTask num {}, Epoch num {}, Train Accuracy: {:.2f}% Test Accuracy: {:.2f}%\n'.format(
                task, epoch, train_acc, test_acc))

        test_accuracies = []

        for i in range(task + 1):
            total = 0

            for data, target in test_loader:
                data, target = jnp.array(data.numpy().reshape((-1, image_size))), jnp.array(target.numpy())
                data = data[:, permutations[i]]

                n_correct = eval_step(model, pytrees, data, target, config.train_mc_iters)
                total += n_correct

            test_acc = 100. * total / (len(test_loader.dataset) * config.train_mc_iters)
            test_accuracies.append(test_acc)
            print('\nTraning task Num: {} Test Accuracy of task {}: {:.2f}%\n'.format(
                task, i, test_acc))

        print(test_accuracies)
        ava_test.append(jnp.mean(test_accuracies))
        return ava_test
