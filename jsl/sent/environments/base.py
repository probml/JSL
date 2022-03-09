import jax.numpy as jnp
from jax import random, jit, nn, vmap

import haiku as hk

import chex
from typing import Callable, List, Tuple
from sklearn.preprocessing import PolynomialFeatures

from jsl.sent.environments.sequential_data_env import SequentialDataEnvironment
from jsl.sent.environments.sequential_torch_env import SequentialTorchEnvironment


def gaussian_sampler(key: chex.PRNGKey, shape: Tuple) -> chex.Array:
    # shape: (num_samples: int, input_dim: int)
    return random.normal(key, shape)

def eveny_spaced_x_sampler(max_val: float, num_samples: int, use_bias=True)->chex.Array:
    X = jnp.linspace(0, max_val, num_samples)
    if use_bias:
        X = jnp.c_[jnp.ones(num_samples), X]
    else:
        X = X.reshape((-1, 1))
    return X

def make_matlab_demo_environment(train_batch_size: int= 1,
                                test_batch_size: int=128):
  # Data from original matlab example
  # https://github.com/probml/pmtk3/blob/master/demos/linregOnlineDemoKalman.m

  max_val, N = 20., 21
  X = eveny_spaced_x_sampler(max_val, N)
  Y = jnp.array([2.4865, -0.3033, -4.0531, -4.3359,
                -6.1742, -5.604, -3.5069, -2.3257,
                -4.6377, -0.2327, -1.9858, 1.0284,
                -2.264, -0.4508, 1.1672, 6.6524,
                4.1452, 5.2677, 6.3403, 9.6264, 14.7842]).reshape((-1, 1))
    
  env = SequentialDataEnvironment(X, Y,
                                X, Y,
                                train_batch_size, test_batch_size,
                                classification=False)

  return env


def make_random_poly_regression_environment(key: chex.PRNGKey,
                                            degree: int,
                                            ntrain: int,
                                            ntest: int,
                                            obs_noise: float=0.01,
                                            train_batch_size: int=1,
                                            test_batch_size: int=128,
                                            x_generator: Callable=gaussian_sampler):
  nsamples = ntrain + ntest
  X = x_generator(key, (nsamples, 1))

  poly = PolynomialFeatures(degree)
  Phi = jnp.array(poly.fit_transform(X), dtype=jnp.float32)

  D = Phi.shape[-1]
  w = random.normal(key, (D, 1))

  if obs_noise > 0.0:
    noise = random.normal(key, (nsamples, 1)) * obs_noise
  Y = Phi @ w + noise
  
  X_train = X[:ntrain]
  X_test = X[ntrain:]
  y_train = Y[:ntrain]
  y_test = Y[ntrain:]
  
  env = SequentialDataEnvironment(X_train, y_train,
                                X_test, y_test,
                                train_batch_size, test_batch_size,
                                classification=False)
  
  return env


def make_random_linear_regression_environment(key: chex.PRNGKey,
                                            nfeatures: int,
                                            ntargets: int,
                                            ntrain: int,
                                            ntest: int,
                                            bias: float=0.0,
                                            obs_noise: float=0.0,
                                            train_batch_size: int=1,
                                            test_batch_size: int=128,
                                            x_generator: Callable=gaussian_sampler):
    # https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/datasets/_samples_generator.py#L506

    nsamples = ntrain + ntest
    # Randomly generate a well conditioned input set
    x_key, w_key, noise_key = random.split(key, 3) 

    X = x_generator(x_key, (nsamples, nfeatures))

    # Generate a ground truth model with only n_informative features being non
    # zeros (the other features are not correlated to y and should be ignored
    # by a sparsifying regularizers such as L1 or elastic net)
    ground_truth = 100 * random.normal(w_key,(nfeatures, ntargets))

    Y = jnp.dot(X, ground_truth) + bias

    # Add noise
    if obs_noise > 0.0:
        Y += obs_noise * random.normal(noise_key, size=Y.shape)

    X_train = X[:ntrain]
    X_test = X[ntrain:]
    y_train = Y[:ntrain]
    y_test = Y[ntrain:]

    env = SequentialDataEnvironment(X_train, y_train,
                                    X_test, y_test,
                                    train_batch_size, test_batch_size,
                                    classification=False)
    return env


def make_mlp(key: chex.PRNGKey,
            nfeatures: int,
            ntargets: int,
            temperature: float,
            hidden_layer_sizes: List[int]):

    assert hidden_layer_sizes != []

    # Generating the logit function
    def net_fn(x: chex.Array):
        """Defining the generative model MLP."""            
        hidden = hidden_layer_sizes[0]
        y = hk.Linear(
            output_size=hidden,
            b_init=hk.initializers.RandomNormal(1./jnp.sqrt(nfeatures)),
        )(x)
        y = nn.relu(y)

        for hidden in hidden_layer_sizes[1:]:
            y = hk.Linear(hidden)(y)
            y = nn.relu(y)
        return hk.Linear(ntargets)(y)
  
    transformed = hk.without_apply_rng(hk.transform(net_fn))

    dummy_input = jnp.zeros([1, nfeatures])
    params = transformed.init(key, dummy_input)

    assert temperature > 0.0
    
    def forward(x: chex.Array):
        return transformed.apply(params, x) / temperature

    y_predictor = jit(forward)

    return y_predictor


def make_classification_mlp_environment(key: chex.PRNGKey,
                                        nfeatures: int,
                                        ntargets: int,
                                        ntrain: int,
                                        ntest: int,
                                        temperature: float,
                                        hidden_layer_sizes: List[int],
                                        train_batch_size=1,
                                        test_batch_size=128,
                                        x_generator=gaussian_sampler):

    x_key, y_key = random.split(key)
    y_predictor = make_mlp(y_key,
                    nfeatures,
                    ntargets,
                    temperature,
                    hidden_layer_sizes)

    nsamples = ntrain + ntest
    # Generates training data for given problem
    X = x_generator(x_key, (nsamples, nfeatures))

    # Generate environment function across x_train
    train_logits = y_predictor(X)  # [n_train, n_class]
    train_probs = nn.softmax(train_logits, axis=-1)

    # Generate training data.
    def sample_output(probs: chex.Array, key: chex.PRNGKey) -> chex.Array:
        return random.choice(key, ntargets, shape=(1,), p=probs)
    
    y_keys = random.split(y_key, nsamples)

    Y = vmap(sample_output)(train_probs, y_keys)

    X_train = X[:ntrain]
    X_test = X[ntrain:]
    y_train = Y[:ntrain]
    y_test = Y[ntrain:]

    env = SequentialDataEnvironment(X_train, y_train,
                                    X_test, y_test,
                                    train_batch_size, test_batch_size,
                                    classification=True)
    return env


def make_regression_mlp_environment(key: chex.PRNGKey,
                                    nfeatures: int,
                                    ntargets: int,
                                    ntrain: int,
                                    ntest: int,
                                    temperature: float,
                                    hidden_layer_sizes: List[int],
                                    train_batch_size: int=1,
                                    test_batch_size: int=128,
                                    x_generator=gaussian_sampler):

    x_key, y_key = random.split(key)
    y_predictor = make_mlp(y_key,
                    nfeatures,
                    ntargets,
                    temperature,
                    hidden_layer_sizes)

    nsamples = ntrain + ntest
    # Generates training data for given problem
    X = x_generator(x_key, (nsamples, nfeatures))

    # Generate environment function across x_train
    Y = y_predictor(X)  # [n_train, output_dim]
    
    X_train = X[:ntrain]
    X_test = X[ntrain:]
    y_train = Y[:ntrain]
    y_test = Y[ntrain:]

    env = SequentialDataEnvironment(X_train, y_train,
                                    X_test, y_test,
                                    train_batch_size, test_batch_size,
                                    classification=False)
    return env
    

def make_environment_from_torch_dataset(dataset: Callable,
                                        classification: bool,
                                        train_batch_size: int = 1,
                                        test_batch_size: int = 128):
    env = SequentialTorchEnvironment(dataset,
                                    train_batch_size,
                                    test_batch_size,
                                    classification)
    return env