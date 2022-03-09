import chex

from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def predict(self, x:chex.Array) -> chex.Array:
        "Predicts the y values given x."

    @abstractmethod
    def update(self, x_train: chex.Array,
                     y_train: chex.Array):
        "Updates its belief state given the training data."
    
    @abstractmethod
    def reset(self):
        "Resets its belief state."

