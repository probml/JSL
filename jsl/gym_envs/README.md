# THIS CODE IS DEPRECATED (as of 2022-03-09)

# Sequential Learning Gym

We extend the [neural testbed](https://github.com/deepmind/neural_testbed) from Deepmind to handle online or continual supervised learning. Environments are implemented as custom OpenAI Gym environments. At the t'th step, the environment creates a data sample from p_t(X,Y); the agent updates its beliefs
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
 
