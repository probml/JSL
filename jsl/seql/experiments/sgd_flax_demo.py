import jax.numpy as jnp
from jax import random

import optax

import torchvision

from functools import partial

from jsl.seql.environments.base import make_environment_from_torch_dataset
from jsl.seql.agents.sgd_agent import sgd_agent
from jsl.seql.utils import train, classification_loss
from jsl.seql.experiments.experiment_utils import LeNet5

nclasses = 10
model = LeNet5(nclasses)

def loss_fn(params, inputs, labels, predict_fn):
  logprobs = predict_fn(params, inputs)
  loss = classification_loss(labels, logprobs)
  return loss

def callback_fn(**kwargs):
  
  logprobs = model.apply(kwargs["belief_state"].params, kwargs["X_train"])
  y_pred = jnp.argmax(logprobs, axis=-1)
  train_acc = jnp.mean(y_pred == kwargs["Y_train"])

  logprobs = model.apply(kwargs["belief_state"].params, kwargs["X_train"])
  y_pred = jnp.argmax(kwargs["preds"][0], axis=-1)
  test_acc = jnp.mean(y_pred == kwargs["Y_test"])
  print("Loss: ", kwargs["info"].loss)
  print(f"Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")

def main():
    key = random.PRNGKey(0)

    dataset = torchvision.datasets.MNIST
    classification = True
    batch_size = 64
    env = make_environment_from_torch_dataset(dataset,
                                              classification,
                                              batch_size,
                                              batch_size)
    image_size = (28, 28)
    batch = jnp.ones((1, *image_size))
    partial_loss = partial(loss_fn, predict_fn=model.apply)
    
    agent = sgd_agent(partial_loss,
                      model.apply,
                      optimizer=optax.adam(1e-3),
                      obs_noise=0.)

    variables = model.init(key, batch)
    belief = agent.init_state(variables)
    nsteps = 60 * 10
    _ = train(belief, agent, env, nsteps, callback_fn)
                    

if __name__=="__main__":
    main()