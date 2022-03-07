# SNB

In this repository, the neural testbed benchmark is extended to the online or continual learning setting. Environments are implemented as custom OpenAI Gym environmennts.

## Agents

- Embedded Extended Kalman Filter
- Kalman filter
- Multi Layer Perceptron

## Environments

There are two different environment types which stand for the type of supervised learning probleem. They not only produce synthetic data given the distribution of training data and test data but also use any available dataset.

- Classification Environment
- Regression Environment

In order to use your own dataset, 

1. You should define two functions which returns `x_train_generator` and  `x_test_generator`, respectively. Both of them should look like 

```
def make_x_sampler(input_dim: int):
    def x_train_sampler(key: chex.PRNGKey, num_samples: int):
        ...
        return X
    return x_sampler
```

2.  You should define sample_fn as follows:

```
def sample_fn(apply_fn: Callable,
                         x_generator: Callable,
                         num_train: int,
                         key: chex.PRNGKey):
                
  x_train = x_generator(key,  num_train)
  y_train = ...
   ...
  data = (x_train, y_train)
  return data
```
Note that `apply_fn` is usually used for creating synthetic dataset.

## How to run

You can either create your own config file or use the predefined ones. Then, you should run

```
python3 -m jsl.gym_envs.run --config <path-of-config-file>
```
