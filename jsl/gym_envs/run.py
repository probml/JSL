# Main function

from gym.envs.registration import register

import gym

import jax.numpy as jnp
from jax import random

from absl import flags
from absl import app

from ml_collections import config_flags

from envs.base import make_mlp_logit_fn, make_linear_logit_fn
from envs.base import make_poly_fit_fn, make_gaussian_sampler
from agents.kalman_filter_regression import KalmanFilterReg

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)

register(
    id="seqcls-v0",
    entry_point="envs:ClassificationEnv",
    max_episode_steps=320)

register(
    id="seqreg-v0",
    entry_point="envs:RegressionEnv",
    max_episode_steps=320)

problems_to_id = {
    "classification": "seqcls-v0",
    "regression": "seqreg-v0"
}

model_to_gen_fns = {
    "MLP": make_mlp_logit_fn,
    "Linear": make_linear_logit_fn,
    "Poly": make_poly_fit_fn
}

'''env = gym.make('OnlineLearning-RandomMLP’, ndim=2, nclasses=2, temperature=01,
nsteps=100, ntrain_per_step=10, ntest_per_step=1000,
test_input_generator=gauss2d)'''


# agent = online_learner_sgd_agent(optimizer=optax.(), buffer_size = 100)

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    config = FLAGS.config

    if not config.problem.lower() or not config.env_model in model_to_gen_fns:
        raise TypeError("Problem type or environment model is not valid.")

    key = random.PRNGKey(config.seed)

    id = problems_to_id[config.problem.lower()]
    generate_fn = model_to_gen_fns[config.env_model]
    fit_key, key = random.split(key)
    fit_fn = generate_fn(config.prior_knowledge.input_dim,
                         config.prior_knowledge.temperature,
                         config.prior_knowledge.hidden,
                         config.prior_knowledge.num_classes,
                         fit_key)
    x_train_generator = make_gaussian_sampler(config.prior_knowledge.input_dim)
    x_test_generator = make_gaussian_sampler(config.prior_knowledge.input_dim)

    env_key, key = random.split(key)
    register(id="seqreg-v0", entry_point="envs:RegressionEnv",
             max_episode_steps=320,
             kwargs={"fit_fn": fit_fn,
                     "x_train_generator": x_train_generator,
                     "x_test_generator": x_test_generator,
                     "prior_knowledge": config.prior_knowledge,
                     "train_batch_size": config.train_batch_size,
                     "test_batch_size": config.test_batch_size,
                     "num_steps": config.nsteps,
                     "key": env_key}
             )
    env = gym.make(id)

    mu0 = jnp.zeros(config.prior_knowledge.input_dim)
    Sigma0 = jnp.eye(config.prior_knowledge.input_dim) * 10.
    F = jnp.eye(config.prior_knowledge.input_dim)
    Q, R = 0, 1
    agent = KalmanFilterReg(mu0, Sigma0, F, Q, R)
    rewards_per_trial = []

    for episode in range(config.ntrials):
        rewards = []
        # agent.reset(next(key))
        reset_key, key = random.split(key)
        obs = env.reset()  # initial train, test split
        agent.update(obs["X_train"], obs["y_train"])
        Ypred = agent.predict(obs["X_test"])

        for t in range(config.nsteps):  # online learning
            obs, reward, done, info = env.step(Ypred)
            rewards.append(reward)
            if done:
                break
            agent.update(obs["X_train"], obs["y_train"])
            Ypred = agent.predict(obs["X_test"])

        rewards_per_trial.append(rewards)


if __name__ == '__main__':
    flags.mark_flags_as_required(['config'])
    app.run(main)
