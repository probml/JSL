# Main function
from jax import tree_map
import jax.numpy as jnp

def train(agent, env, nsteps, save_params=True):
    #env.reset()
    #agent.reset()
    params, rewards = {}, []

    for t in range(nsteps):
        X_train, Y_train, X_test, Y_test = env.get_data(t)
        updated_params = agent.update(X_train, Y_train)
        
        if save_params:
                if params:
                    params = tree_map(lambda x, y: jnp.vstack([x, y]),
                                      params, updated_params)
                else:
                    params = updated_params

        Y_pred = agent.predict(X_test)
        reward = env.reward(Y_pred, Y_test)
        print(reward)
        rewards.append(reward)
        
    return params, rewards