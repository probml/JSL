# SNB API sketch

# https://github.com/deepmind/neural_testbed/blob/master/neural_testbed/experiments/experiment.py

from jax import random, vmap

def run_multi_step(agent, env, num_steps, key):
  agent =  agent.init(env.prior_knowledge, key)
  loss_seq = []
  for t in range(num_steps):
      (X_train, y_train) = env.get_train_data(t)
      agent = agent.update_beliefs(X_train, y_train)
      loss_seq[t] = evaluate_beliefs(env, agent, t)
  return loss_seq
      

def evaluate_beliefs(env, agent, t, num_belief_samples=1000, num_test_samples=1000, tau=10):
  for n in range(num_test_samples):
        (X_test, y_test) = env.get_test_data(t, tau)
        prob_true = one_hot(y_test) 
        for k in range(num_belief_samples):
          pred_fn = agent.sample_posterior_predictive()
          prob_pred = pred_fn(X_test) # tau * num_classes
          cross_ent[k] = sum(prob_true * log(prob_pred), 1).mean() # sum over classes, mean over tau
        kl[n] = cross_ent.mean() # average over belief samples k
  loss = kl.mean() # average over test samples n
  return loss


# https://github.com/deepmind/neural_testbed/blob/master/neural_testbed/leaderboard/sweep.py
def classification_2d_sweep(num_seed: int = 10,
                            initial_seed: int = 0) -> Dict[str, ProblemConfig]:
  """Generate hyperparameter sweep for 2d classification problems.

  Args:
    num_seed: number of seeds per configuratioon of other hyperparameters.
    initial_seed: initial value of the seed.
  Returns:
    Mapping problem_id: gp_settings (for use in gp_load).
  """
  configs = []
  # TODO(author2): convert to itertools
  for tau in [1, 10]:
    seed = initial_seed
    for num_train in [1, 3, 10, 30, 100, 300, 1000]:
      for temperature in [0.01, 0.1, 0.5]:
        for unused_seed_inc in range(num_seed):
          seed += 1
          prior_knowledge = base.PriorKnowledge(
              input_dim=2,
              num_train=num_train,
              num_classes=2,  # Currently fixed and not part of the configs.
              tau=tau,
              layers=2,
              temperature=temperature,
          )

          configs.append(ProblemConfig(prior_knowledge, seed))
  return {f'classification_2d{SEPARATOR}{i}': v
          for i, v in enumerate(configs)}

CLASSIFICATION_2D = tuple(classification_2d_sweep().keys())

def run_multi_env(agent, num_steps, key):
  for config, e in enumerate(CLASSIFICATION_2D):
    env =  RandomMLPEnvironment(config)
    loss_trace = run_multi_step(agent, env, num_steps, key)
    total_loss[e] = loss_trace.sum()
  return total_loss.mean(), total_loss.std()


def make_2layer_mlp_logit_fn(
    input_dim: int,
    temperature: float,
    hidden: int,
    num_classes: int,
    key: chex.PRNGKey,
) -> class_env.LogitFn:
  """Factory method to create a generative model around a 2-layer MLP."""

  # Generating the logit function
  def net_fn(x: chex.Array) -> chex.Array:
    """Defining the generative model MLP."""
    y = hk.Linear(
        output_size=hidden,
        b_init=hk.initializers.RandomNormal(1./jnp.sqrt(input_dim)),
    )(x)
    y = jax.nn.relu(y)
    y = hk.Linear(hidden)(y)
    y = jax.nn.relu(y)
    return hk.Linear(num_classes)(y)

  transformed = hk.without_apply_rng(hk.transform(net_fn))
  dummy_input = jnp.zeros([1, input_dim])
  params = transformed.init(key, dummy_input)
  
  def forward(x: chex.Array) -> chex.Array:
    return transformed.apply(params, x) / temperature
  
  logit_fn = jax.jit(forward)

  return logit_fn


@dataclasses.dataclass(frozen=True)
class PriorKnowledge:
  """What an agent knows a priori about the problem."""
  input_dim: int
  num_train: int
  tau: int
  num_classes: int = 1
  layers: Optional[int] = None
  noise_std: Optional[float] = None
  temperature: Optional[float] = None
  extra: Optional[Dict[str, Any]] = None

#https://github.com/deepmind/neural_testbed/blob/master/neural_testbed/generative/classification_envlikelihood.py

class RandomMLPEnvironment:
  def __init__(self, key, input_dim = 2, num_classes = 2, temperature=0.1):
    
    self.key = key
    self.input_dim = input_dim
    
    self.logit_fn = make_2layer_mlp_logit_fn(input_dim,
                                            temperature,
                                            hidden,
                                            num_classes,
                                            next(self.key))


  def sample_output(self, key, probs):
    num_samples = probs.shape[0]
    keys = random.split(key, num_samples)
    return vmap(random.choice, in_axes(0, 0))(keys, probs)

  def get_train_data(self, t, num_samples=10): 
    # return X, Y for minibatch at each step
    X =  random.normal(next(self.key), [num_samples, self.input_dim])
    probs = vmap(self.logit_fn)(X)
    Y = self.sample_output(next(self.key), probs)
    return X, Y

  def get_test_data(self, t, num_samples=10):
    # sample fresh data from true model
    return get_train_data(t, num_samples)

  def prior_knowledge(self):
    return self.input_dim, self.num_classes


class SGDMLPEnsembleAgent:
    def __init__(self, prior_knowledge, key, hidden_sizes = [50,50], l2_weight_decay=1,
             nensembles = 1, step_size = 0.01):
        self.models = [MLP(hidden_sizes, l2_weight_decay) for n in range(nensembles)]
        self.opt = optax.SGD(step_size)

  # https://github.com/deepmind/enn/blob/master/enn/supervised/sgd_experiment.py
    def update_beliefs(self, X, Y):
        # incrementally fit model on new data
        # we apply the same update to all ensemble members
        # (Could use bootstrap sampling)
        self.models = [self.opt.update(m, X, Y) for m in self.models]


    def sample_posterior_predictive(self, nsamples):
        # sample from belief state
        # Return a set of functions x->logits
        idx = jax.random.uniform(1/self.nensembles, nsamples)
        pred_fns = [self.models[idx].apply]
        return pred_fns
    

