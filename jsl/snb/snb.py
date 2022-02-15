# SNB API sketch


# https://github.com/deepmind/neural_testbed/blob/master/neural_testbed/experiments/experiment.py

def run_multi_step(agent: testbed_base.TestbedAgentSeq,
        problem: testbed_base.TestbedProblemSeq,
        num_steps: Int,
        key: RNG) -> List[testbed_base.ENNQuality]:
  """Run an agent on a given testbed problem for T steps."""
  agent =  agent(problem.prior_knowledge, key)
  quality = []
  for t in range(num_steps):
      data = problem.get_train_data(t)
      agent = agent.update_beliefs(data)
      enn_sampler = agent.sample_posterior_predictive()
      quality.append(problem.evaluate_quality(enn_sampler))
  return quality

# https://github.com/deepmind/neural_testbed/blob/master/neural_testbed/base.py

class TestbedAgentSeq(typing_extensions.Protocol):
    """An interface for specifying a sequential testbed agent."""

    def __call__(self,  prior: PriorKnowledge, key):
        """Sets up a training procedure given ENN prior knowledge."""
        pass

    def update_beliefs(data):
        # incrementally fit model on new data
        pass

    def sample_posterior_predictive() -> EpistemicSampler:
        # sample from belief state
        # Same as current TestbedAgent.cal() function
        pass



class TestbedProblemSeq(abc.ABC):
  """An interface for specifying a sequential testbed problem."""

  @abc.abstractproperty
  def get_train_data(self, t) -> Data:
    """Access training data for t'th step."""

  @abc.abstractproperty
  def prior_knowledge(self) -> PriorKnowledge:
    """Information describing the problem instance."""

  @abc.abstractmethod
  def evaluate_quality(self, enn_sampler: EpistemicSampler) -> ENNQuality:
    """Evaluate the quality of a posterior sampler."""



#####################################################
# https://github.com/deepmind/neural_testbed/blob/master/neural_testbed/leaderboard/sweep.py




# https://github.com/deepmind/neural_testbed/blob/master/neural_testbed/leaderboard/load.py




def problem_from_config(
    problem_config: sweep.ProblemConfig) -> testbed_base.TestbedProblem:
  """Returns a testbed problem given a problem config."""
  assert problem_config.prior_knowledge.num_classes > 0

  if problem_config.prior_knowledge.num_classes > 1:
    return _load_classification(problem_config)
  else:
    return #_load_regression(problem_config)

# we ignore shift_config
def _load_classification(
    problem_config: sweep.ProblemConfig,
) -> likelihood.SampleBasedTestbed:
  """Loads a classification problem from problem_config, optional shift_config."""
  rng = hk.PRNGSequence(problem_config.seed)
  prior_knowledge = problem_config.prior_knowledge
  input_dim = prior_knowledge.input_dim

  logit_fn = generative.make_2layer_mlp_logit_fn(
      input_dim=input_dim,
      temperature=prior_knowledge.temperature,
      hidden=50,
      num_classes=prior_knowledge.num_classes,
      key=next(rng),
  )
    # skip shift_config
  data_sampler = generative.ClassificationEnvLikelihood(
      logit_fn=logit_fn,
      x_train_generator=generative.make_gaussian_sampler(input_dim),
      x_test_generator=problem_config.test_distribution(input_dim),
      num_train=prior_knowledge.num_train,
      key=next(rng),
      override_train_data=override_train_data,
      tau=prior_knowledge.tau,
  )
  return likelihood.SampleBasedTestbed(
      data_sampler=data_sampler,
      sample_based_kl=make_categorical_kl_estimator(problem_config, next(rng)),
      prior_knowledge=prior_knowledge,
  )