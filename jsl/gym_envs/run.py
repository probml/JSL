
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
                  "mlp": make_mlp_apply_fn,
                  "linear": make_mlp_apply_fn,
                  "poly": make_poly_fit_fn
                 }

agents = {
         "KalmanFilter": KalmanFilterReg,
         "EEKF": EmbeddedExtendedKalmanFilter
         }


def main(config: ml_collections.ConfigDict):

    if not config.env.problem_type.lower() in problems_to_id:
        raise TypeError("Environment model is not valid.")

    key = random.PRNGKey(config.seed)

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
    rewards_per_trial = []
    params_per_trial = []

    for episode in range(config.ntrials): 
        rewards, params = [], {}
        # agent.reset(next(key))
        reset_key, key = random.split(key)
        obs = env.reset() # initial train, test split
        agent.update(obs["X_train"], obs["y_train"])
        
        Ypred  = agent.predict(obs["X_test"])

        for t in range(config.nsteps): # online learning
            obs, reward, done, info = env.step(Ypred)
            print(reward)
            rewards.append(reward)
            
            if done:
                break
            
            updated_params = agent.update(obs["X_train"], obs["y_train"])
            
            if config.save_params:
                if params:
                    params = tree_map(lambda x, y: jnp.vstack([x, y]),
                                      params, updated_params)
                else:
                    params = updated_params

            Ypred  = agent.predict(obs["X_test"]) 

        rewards_per_trial.append(rewards)
        
        if config.save_params:
            params_per_trial.append(params)
    
    return params_per_trial, rewards_per_trial


def from_command_line(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    config = FLAGS.config
    main(config)

if __name__ == '__main__':
  flags.mark_flags_as_required(['config'])
  app.run(from_command_line)