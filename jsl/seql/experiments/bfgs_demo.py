# Load packages
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
from matplotlib import pyplot as plt

from jsl.seql.agents.bfgs_agent import bfgs_agent
from jsl.seql.environments.sequential_data_env import SequentialDataEnvironment
from jsl.seql.utils import train

def model_fn(params, x):
    return jnp.dot(x, params[:-1])


# Construct LLF
def llf(params, inputs, outputs, model_fn):
    s2hat = jnp.exp(params[-1])
    xb = model_fn(params, inputs)
    ll = norm.logpdf(jnp.squeeze(outputs)- xb,
                     0,
                     jnp.sqrt(s2hat))
    return -jnp.mean(ll)


def main():
    # Set initial setup of parameters
    B = jnp.array([4., 2., 10])
    N = 400

    key = random.PRNGKey(0)
    u_key, x1_key = random.split(key)

    # Construct x and y following normal linear model
    s2 = 10.
    u = jnp.sqrt(s2)* random.normal(u_key, (N, ))

    x0 = jnp.ones((N,))
    x1 = random.normal(x1_key, (N,))

    X = jnp.stack((x0, x1), axis=1)
    y = (jnp.dot(X, B[:-1])+ u).reshape((-1, 1))

    initial_value = jnp.array([0.1, 10, 0.1])

    agent = bfgs_agent(llf, model_fn)
    belief = agent.init_state(initial_value)

    batch_size = 100
    env = SequentialDataEnvironment(X, y, X, y,
                                    train_batch_size=batch_size,
                                    test_batch_size=1,
                                    classification=False)
    belief, _ = train(belief, agent, env, nsteps=N //batch_size)
    
    ypred = agent.predict(belief, X)

    fig, ax = plt.subplots()
    ax.scatter(X[:, 1], y, alpha=0.1)

    indices = jnp.argsort(X[:, 1])
    X = X[indices]
    ypred = ypred[indices]

    ax.plot(X[:, 1], ypred)
    ax.set_title("BFGS-Linear Regression")
    plt.savefig("bfgs_linreg.png")

if __name__=="__main__":
    main()
