import jax.numpy as jnp
from jax import random, nn
from sklearn.preprocessing import PolynomialFeatures

from jsl.experimental.seql.agents.sgd_agent import sgd_agent
from jsl.experimental.seql.environments.base import make_random_poly_classification_environment
from jsl.experimental.seql.experiments.plotting import plot_classification_2d
from jsl.experimental.seql.utils import binary_cross_entropy, train

def callback_fn(agent, env, model_fn, obs_noise, degree, **kwargs):
    belief = kwargs["belief_state"]
    mu, sigma = belief.params, None

    logprobs, _ = kwargs["preds"]
    y_test = jnp.squeeze(kwargs["Y_test"])
    predictions = jnp.where(logprobs > jnp.log(0.5), 1, 0)
    print("Accuracy: ", jnp.mean(jnp.squeeze(predictions)==y_test))

    poly = PolynomialFeatures(degree)
    grid = poly.fit_transform(jnp.mgrid[-3:3:100j, -3:3:100j].reshape((2, -1)).T)
    grid_preds = agent.predict(belief, grid)

    filename = "poly_logreg_sgd_ppd"
    timesteps = [5, 10, 15, 75]
    plot_classification_2d(env,
                                    mu,
                                    sigma,
                                    obs_noise,
                                    timesteps,
                                    grid,
                                    grid_preds,
                                    filename,
                                    model_fn=model_fn,
                                    **kwargs)

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
    nfeatures, nclasses = 2, 2
    env = make_random_poly_classification_environment(key,
                                                  degree,
                                                  ntrain,
                                                  ntest,
                                                  nfeatures=nfeatures,
                                                  nclasses=nclasses)
    

   
    obs_noise = 0.01
    buffer_size = 1
    input_dim = env.X_train.shape[-1]
    agent = sgd_agent(loss_fn,
                      model_fn,
                      obs_noise=obs_noise,
                      buffer_size=buffer_size)

    params = jnp.zeros((input_dim, 1))
    belief = agent.init_state(params)

    nsteps = 100
    partial_callback = lambda **kwargs: callback_fn(agent,
                                                    env,
                                                    model_fn,
                                                    obs_noise,
                                                    degree,
                                                    **kwargs)

    _, unused_rewards = train(belief,
                              agent,
                              env,
                              nsteps=nsteps,
                              callback=partial_callback)
                 

    

if __name__ == "__main__":
    main()