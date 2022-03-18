import jax.numpy as jnp
from jax import random, nn
from jsl.experimental.seql.agents.sgd_agent import sgd_agent

from jsl.experimental.seql.environments.base import make_random_poly_classification_environment
from jsl.experimental.seql.utils import binary_cross_entropy, train

def callback_fn(**kwargs):
    logprobs, _ = kwargs["preds"]
    y_test = jnp.squeeze(kwargs["Y_test"])
    predictions = jnp.where(logprobs > jnp.log(0.5), 1, 0)
    print("Accuracy: ", jnp.mean(jnp.squeeze(predictions)==y_test))

def model_fn(params, x):
    return nn.log_sigmoid(x @ params)

def loss_fn(params, x, y, model_fn):
    logprobs = model_fn(params, x)
    return binary_cross_entropy(y, logprobs)

def main():
    key = random.PRNGKey(0)
    degree = 3
    ntrain = 200  # 80% of the data
    ntest = 50  # 20% of the data
    input_dim, nclasses = degree + 1, 2

    env = make_random_poly_classification_environment(key,
                                                  degree,
                                                  ntrain,
                                                  ntest,
                                                  nclasses=nclasses)
    

   
    obs_noise = 0.01
    buffer_size = 1
    agent = sgd_agent(loss_fn,
                      model_fn,
                      obs_noise=obs_noise,
                      buffer_size=buffer_size)

    params = jnp.zeros((input_dim, 1))
    belief = agent.init_state(params)

    nsteps = 100
    _, unused_rewards = train(belief,
                              agent,
                              env,
                              nsteps=nsteps,
                              callback=callback_fn)
                 

    

if __name__ == "__main__":
    main()