import jax.numpy as jnp
from jax import random, tree_leaves, tree_map
from jax import nn

from functools import partial
from matplotlib import pyplot as plt

import optax

from jsl.experimental.seql.agents.bfgs_agent import bfgs_agent
from jsl.experimental.seql.agents.blackjax_nuts_agent import blackjax_nuts_agent
from jsl.experimental.seql.agents.lbfgs_agent import lbfgs_agent
from jsl.experimental.seql.agents.scikit_log_reg_agent import scikit_log_reg_agent
from jsl.experimental.seql.agents.sgd_agent import sgd_agent
from jsl.experimental.seql.agents.sgmcmc_sgld_agent import sgld_agent
from jsl.experimental.seql.environments.base import make_random_poly_classification_environment
from jsl.experimental.seql.experiments.bimodal_reg_demo import loglikelihood_fn
from jsl.experimental.seql.experiments.experiment_utils import run_experiment
from jsl.experimental.seql.experiments.plotting import colors
from jsl.experimental.seql.utils import cross_entropy_loss


def model_fn(w, x):
    return nn.log_softmax(x @ w, axis=-1)


def logprior_fn(params, strength=0.2):
    leaves = tree_leaves(params)
    return -sum(tree_map(lambda x: jnp.sum(x ** 2), leaves)) * strength


def loglikelihood_fn(params, x, y, model_fn):
    logprobs = model_fn(params, x)
    return -cross_entropy_loss(y, logprobs)


def logjoint_fn(params, inputs, outputs, model_fn, strength=0.2):
    return loglikelihood_fn(params, inputs, outputs, model_fn)  # + logprior_fn(params, strength) / len(inputs)


def neg_logjoint_fn(params, inputs, outputs, model_fn, strength=0.2):
    return -logjoint_fn(params, inputs, outputs, model_fn, strength)


def print_accuracy(logprobs, ytest):
    ytest_ = jnp.squeeze(ytest)
    predictions = jnp.where(logprobs > jnp.log(0.5), 1, 0)
    print("Accuracy: ", jnp.mean(jnp.argmax(predictions, axis=-1) == ytest_))


losses = []


def callback_fn(agent, env, agent_name, **kwargs):
    global losses
    if kwargs["t"] == 0:
        losses = []
    belief = kwargs["belief_state"]
    nfeatures = kwargs["nfeatures"]
    out = 1

    inputs = env.X_test.reshape((-1, nfeatures))
    outputs = env.y_test.reshape((-1, out))
    predictions, _ = agent.predict(belief, inputs)
    loss = jnp.mean(jnp.power(predictions - outputs, 2))
    losses.append(loss)
    if kwargs["t"] == 9:
        ax = kwargs["ax"]
        ax.plot(losses, color=colors[agent_name])
        ax.set_title(agent_name.upper())
        plt.tight_layout()
        plt.savefig("asas.png")


def initialize_params(agent_name, **kwargs):
    nfeatures = kwargs["nfeatures"]
    mu0 = random.normal(random.PRNGKey(0), (nfeatures, 2))
    if agent_name == "bayes" or agent_name == "eekf":
        mu0 = jnp.zeros((nfeatures, 2))
        Sigma0 = jnp.eye(nfeatures)
        initial_params = (mu0, Sigma0)
    else:
        initial_params = (mu0,)

    return initial_params


def main():
    key = random.PRNGKey(0)

    degree = 3
    ntrain, ntest = 100, 100
    batch_size = 10
    nsteps = 10
    nfeatures, nclasses = 2, 2

    env_key, nuts_key, sgld_key = random.split(key, 3)
    obs_noise = 1.
    env = lambda batch_size: make_random_poly_classification_environment(env_key,
                                                                         degree,
                                                                         ntrain,
                                                                         ntest,
                                                                         nfeatures=nfeatures,
                                                                         nclasses=nclasses,
                                                                         obs_noise=obs_noise,
                                                                         train_batch_size=batch_size,
                                                                         test_batch_size=batch_size,
                                                                         shuffle=False)

    buffer_size = ntrain

    input_dim = 10
    optimizer = optax.adam(1e-2)

    tau = 1.
    strength = obs_noise / tau
    partial_objective_fn = partial(neg_logjoint_fn, strength=strength)

    nepochs = 20

    sgd = sgd_agent(partial_objective_fn,
                    model_fn,
                    optimizer=optimizer,
                    obs_noise=obs_noise,
                    nepochs=nepochs,
                    buffer_size=buffer_size)

    batch_sgd = sgd_agent(partial_objective_fn,
                          model_fn,
                          optimizer=optimizer,
                          obs_noise=obs_noise,
                          buffer_size=buffer_size,
                          nepochs=nepochs * nsteps)

    nsamples, nwarmup = 500, 300
    nuts = blackjax_nuts_agent(nuts_key,
                               loglikelihood_fn,
                               model_fn,
                               nsamples=nsamples,
                               nwarmup=nwarmup,
                               obs_noise=obs_noise,
                               buffer_size=buffer_size)

    batch_nuts = blackjax_nuts_agent(nuts_key,
                                     loglikelihood_fn,
                                     model_fn,
                                     nsamples=nsamples * nsteps,
                                     nwarmup=nwarmup,
                                     obs_noise=obs_noise,
                                     buffer_size=buffer_size)

    scikit_agent = scikit_log_reg_agent(buffer_size=buffer_size)

    partial_logprob_fn = partial(loglikelihood_fn,
                                 model_fn=model_fn)
    dt = 1e-5
    sgld = sgld_agent(sgld_key,
                      partial_logprob_fn,
                      logprior_fn,
                      model_fn,
                      dt=dt,
                      batch_size=batch_size,
                      nsamples=nsamples,
                      obs_noise=obs_noise,
                      buffer_size=buffer_size)

    dt = 1e-5
    batch_sgld = sgld_agent(sgld_key,
                            partial_logprob_fn,
                            logprior_fn,
                            model_fn,
                            dt=dt,
                            batch_size=batch_size,
                            nsamples=nsamples * nsteps,
                            obs_noise=obs_noise,
                            buffer_size=buffer_size)

    bfgs = bfgs_agent(partial_objective_fn,
                      obs_noise=obs_noise,
                      buffer_size=buffer_size)

    lbfgs = lbfgs_agent(partial_objective_fn,
                        obs_noise=obs_noise,
                        history_size=buffer_size)

    agents = {
        "scikit": scikit_agent,
        "sgd": sgd,
        "bfgs": bfgs,
        "lbfgs": lbfgs,
        "nuts": nuts,
        "sgld": sgld
    }

    batch_agents = {
        "sgd": batch_sgd,
        "nuts": batch_nuts,
        "sgld": batch_sgld,
        "bfgs": bfgs,
        "lbfgs": lbfgs,
        "scikit": scikit_agent,
    }
    nrows = len(agents)
    ncols = 1
    run_experiment(agents,
                   env,
                   initialize_params,
                   batch_size,
                   ntrain,
                   nsteps,
                   nrows,
                   ncols,
                   callback_fn=callback_fn,
                   figsize=(6, 8),
                   nfeatures=input_dim,
                   obs_noise=obs_noise,
                   batch_agents=batch_agents,
                   timesteps=list(range(nsteps)),
                   degree=degree)


if __name__ == "__main__":
    main()
