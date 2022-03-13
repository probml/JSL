# Main function

def train(initial_belief_state, agent, env, nsteps, callback=None):
    #env.reset()
    #agent.reset()
    rewards = []
    belief_state = initial_belief_state

    for t in range(nsteps):
        X_train, Y_train, X_test, Y_test = env.get_data(t)
        
        belief_state, info = agent.update(belief_state, X_train, Y_train)
        
        Y_pred = agent.predict(belief_state, X_test)
        reward = env.reward(Y_pred, Y_test)

        if callback:
            if not isinstance(callback, list):
                callback_list = [callback]
            else:
                callback_list = callback

            for f in callback_list:
                f(belief_state=belief_state,
                  info=info,
                  X_train=X_train,
                  Y_train=Y_train,
                  X_test=Y_test,
                  Y_pred=Y_pred,
                  reward=reward)

        print(f"Time {t + 1}, Reward: {reward}")
        rewards.append(reward)
        
    return belief_state, rewards