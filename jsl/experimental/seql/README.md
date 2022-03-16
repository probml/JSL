# Sequential Learning

We extend the [neural testbed](https://github.com/deepmind/neural_testbed) from DeepMind to handle online or continual supervised learning. Environments are implemented as custom OpenAI Gym environments. At the t'th step, the environment creates a data sample from p_t(X,Y); the agent updates its beliefs
about the function, and makes a prediction on a test set; the environment then gives the agent a reward, and the process repeats.

More precisely, the main code looks like this:
```

 obs1 = env.reset() # obs = { "X_train": Ntr*Din, "y_train": Ntr*Dout, "X_test": Nte*Din } 
 agent.update(obs1["X_train"], obs1["y_train"])
 Ypred1  = agent.predict(obs["X_test"])
 obs2, reward1, done, info = env.step(Ypred) # reward1 = loglik(Ypred1, Ytest1)

 agent.update(obs2["X_train"], obs2["y_train"])
 Ypred2  = agent.predict(obs2["X_test"])
 obs3, reward2, done, info = env.step(Ypred2) # reward2 = loglik(Ypred2, Ytest2)
 ...
 ```
 

## Agents

An agent is a likelihood model of the form p(y|x,theta) and an inference algorithm for the posterior p(theta|D(1:t))
Examples:

Models:
- Linear regresson (with fixed basis function): N(y|w' phi(x), sigma^2) where phi(x) is specified.
- Logistic regresson (with fixed basis function): Cat(y|softmax(W' phi(x))),  where phi(x) is specified. 
- MLP
- CNN
-  
Posterior inference algorithms:

- SGD 
- "Deep ensembles" ie SGD on multiple copies of the model
- (Extended) Kalman Filter 
- Sequential VI

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
