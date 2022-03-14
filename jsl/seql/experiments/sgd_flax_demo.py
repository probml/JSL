import jax.numpy as jnp
from jax import random

import optax

import torchvision

from functools import partial

from jsl.seql.environments.base import make_environment_from_torch_dataset
from jsl.seql.agents.sgd_agent import sgd_agent
from jsl.seql.experiments.experiment_utils import LeNet5, cross_entropy_loss
from jsl.seql.utils import train


def callback_fn(**kwargs):
  print("Loss: ", kwargs["info"].loss)

def main():
    key = random.PRNGKey(0)

    dataset = torchvision.datasets.MNIST
    classification = True
    batch_size = 100
    env = make_environment_from_torch_dataset(dataset,
                                              classification,
                                              batch_size,
                                              batch_size)
    nclasses = 10
    image_size = (28, 28)
    model = LeNet5(nclasses)
    batch = jnp.ones((1, *image_size))
    partial_classification_loss = partial(cross_entropy_loss,
                                          predict_fn=model.apply)
    agent = sgd_agent(partial_classification_loss,
                      model.apply,
                      optimizer=optax.adam(1e-2),
                      obs_noise=0.)

    variables = model.init(key, batch)
    belief = agent.init_state(variables)
    nsteps = 40
    _ = train(belief, agent, env, nsteps, callback_fn)
                    

if __name__=="__main__":
    main()