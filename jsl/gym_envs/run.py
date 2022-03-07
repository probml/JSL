# Main function
import gym

import jax.numpy as jnp
from jax import random, tree_map

from absl import flags
from absl import app

from ml_collections import config_flags
import ml_collections
from jsl.gym_envs.agents.eekf import EmbeddedExtendedKalmanFilter

from jsl.gym_envs.envs.base import make_mlp_apply_fn
from jsl.gym_envs.envs.base import make_poly_fit_fn
from jsl.gym_envs.agents.kalman_filter import KalmanFilterReg

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)

problems_to_id = {
    "classification": "seqcls-v0",
    "regression": "seqreg-v0"
}

model_to_gen_fns = {
<<<<<<< HEAD
                  "mlp": make_mlp_apply_fn,
                  "linear": make_mlp_apply_fn,
                  "poly": make_poly_fit_fn
                 }
=======
    "MLP": make_mlp_logit_fn,
    "Linear": make_linear_logit_fn,
    "Poly": make_poly_fit_fn
}
>>>>>>> 2433858c0d42c29c0039940078eae7ce1c26a19c

agents = {
         "KalmanFilter": KalmanFilterReg,
         "EEKF": EmbeddedExtendedKalmanFilter
         }

<<<<<<< HEAD
=======

# agent = online_learner_sgd_agent(optimizer=optax.(), buffer_size = 100)
>>>>>>> 2433858c0d42c29c0039940078eae7ce1c26a19c

def main(config: ml_collections.ConfigDict):

    if not config.env.problem_type.lower() in problems_to_id:
        raise TypeError("Environment model is not valid.")

    key = random.PRNGKey(config.seed)

<<<<<<< HEAD
    create_model_fn = model_to_gen_fns[config.env.model.lower()]
    model_key, key = random.split(key)
    
    prior_knowledge = config.env.prior_knowledge

    apply_fn = create_model_fn(prior_knowledge.input_dim,
                        prior_knowledge.output_dim,
                        prior_knowledge.hidden_layer_sizes,
                        prior_knowledge.temperature,
                        model_key)
    
    x_train_generator = config.env.train_distribution(prior_knowledge.input_dim)
    x_test_generator = config.env.test_distribution(prior_knowledge.input_dim)

    env_key, key = random.split(key)

    id = problems_to_id[config.env.problem_type.lower()]                
    env = gym.make(id,
                apply_fn=apply_fn,
                x_train_generator=x_train_generator,
                x_test_generator=x_test_generator,
                prior_knowledge=prior_knowledge,
                train_batch_size=config.env.train_batch_size,
                test_batch_size=config.env.test_batch_size,
                sample_fn=config.env.sample_fn,
                nsteps=config.nsteps,
                key=env_key
                )
    
    agent = agents[config.agent.model](**config.agent.init_kwargs)
=======
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
>>>>>>> 2433858c0d42c29c0039940078eae7ce1c26a19c
    rewards_per_trial = []
    params_per_trial = []

<<<<<<< HEAD
    for episode in range(config.ntrials): 
        rewards, params = [], {}
=======
    for episode in range(config.ntrials):
        rewards = []
>>>>>>> 2433858c0d42c29c0039940078eae7ce1c26a19c
        # agent.reset(next(key))
        reset_key, key = random.split(key)
        obs = env.reset()  # initial train, test split
        agent.update(obs["X_train"], obs["y_train"])
<<<<<<< HEAD
        
        Ypred  = agent.predict(obs["X_test"])
=======
        Ypred = agent.predict(obs["X_test"])
>>>>>>> 2433858c0d42c29c0039940078eae7ce1c26a19c

        for t in range(config.nsteps):  # online learning
            obs, reward, done, info = env.step(Ypred)
            print(reward)
            rewards.append(reward)
            
            if done:
                break
<<<<<<< HEAD
            
            updated_params = agent.update(obs["X_train"], obs["y_train"])
            
            if config.save_params:
                if params:
                    params = tree_map(lambda x, y: jnp.vstack([x, y]),
                                      params, updated_params)
                else:
                    params = updated_params

            Ypred  = agent.predict(obs["X_test"]) 
=======
            agent.update(obs["X_train"], obs["y_train"])
            Ypred = agent.predict(obs["X_test"])
>>>>>>> 2433858c0d42c29c0039940078eae7ce1c26a19c

        rewards_per_trial.append(rewards)
        
        if config.save_params:
            params_per_trial.append(params)
    
    return params_per_trial, rewards_per_trial


<<<<<<< HEAD
def from_command_line(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    config = FLAGS.config
    main(config)

if __name__ == '__main__':
  flags.mark_flags_as_required(['config'])
  app.run(from_command_line)
=======
if __name__ == '__main__':
    flags.mark_flags_as_required(['config'])
    app.run(main)
>>>>>>> 2433858c0d42c29c0039940078eae7ce1c26a19c
