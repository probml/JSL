# Sequential Learning

We extend the [neural testbed](https://github.com/deepmind/neural_testbed) from DeepMind to handle online or continual supervised learning. 
 

## Agents

An agent is a likelihood model of the form p(y|x,theta) and an inference algorithm for the posterior p(theta|D(1:t))
Examples:

Models:
- Linear regresson (with fixed basis function): N(y|w' phi(x), sigma^2) where phi(x) is specified.
- Logistic regresson (with fixed basis function): Cat(y|softmax(W' phi(x))),  where phi(x) is specified. 
- MLP
- CNN
-  
Posterior inference algorithms:

- SGD 
- "Deep ensembles" ie SGD on multiple copies of the model
- (Extended) Kalman Filter 
- Sequential VI

## Environments

There are two different environment types which stand for the type of supervised learning probleem. They not only produce synthetic data given the distribution of training data and test data but also use any available dataset.

- Classification Environment
- Regression Environment

In order to use your own dataset, 
