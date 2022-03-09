import jax.numpy as jnp
import numpy as np

import torch
import torchvision

import chex

import os

from jsl.sent.environments.sequential_data_env import classification_loss, regression_loss


def collate_fn(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        return type(batch[0])(collate_fn(samples) for samples in zip(*batch))
    else:
        return jnp.asarray(batch)

class TorchDataset(torch.utils.data.Dataset):
    # https://github.com/google/jax/issues/3382
    def __init__(self, dataset: torchvision.datasets, train: bool):
        super().__init__()
        self._data = dataset(os.getcwd(), train, download=True)
        self._data_len = len(self._data)

    def __getitem__(self, index: int):
        img, label = self._data[index]
        return jnp.asarray(np.asarray(img)), label

    def __len__(self):
        return self._data_len

class SequentialTorchEnvironment:
  def __init__(self, dataset: torch.utils.data.Dataset,
              train_batch_size: int,
              test_batch_size: int,
              classification: bool):

    self.train_data = TorchDataset(dataset, train=True)
    self.train_dataloader = torch.utils.data.DataLoader(self.train_data,
                                         batch_size=train_batch_size, 
                                         collate_fn=collate_fn,
                                         num_workers=0,)

    self.test_data = TorchDataset(dataset, train=False)
    self.test_dataloader = torch.utils.data.DataLoader(self.test_data,
                                         batch_size=test_batch_size, 
                                         collate_fn=collate_fn,
                                         num_workers=0,)
    
    self.train_data_iterator = iter(self.train_dataloader)
    self.test_data_iterator = iter(self.test_dataloader)
    
    if classification:
      self.loss_fn = classification_loss
    else:
      self.loss_fn = regression_loss

  def get_data(self, t: int):
    # iterate over dataset
    # alternatively you could use while(True) 
    try:
        X_train, y_train = next(self.train_data_iterator) 
        X_test, y_test = next(self.test_data_iterator)
    except StopIteration:
        # StopIteration is thrown if dataset ends
        # reinitialize data loader
        # https://discuss.pytorch.org/t/infinite-dataloader/17903/7
        self.train_data_iterator = iter(self.train_dataloader)
        self.test_data_iterator = iter(self.test_dataloader)

        X_train, y_train = next(self.train_data_iterator) 
        X_test, y_test = next(self.test_data_iterator)
        
    return X_train, y_train, X_test, y_test

  def reward(self, y_pred: chex.Array, y_test: chex.Array):
    loss = self.loss_fn(y_pred, y_test)
    return loss