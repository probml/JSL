from jax import random, vmap
import jax.numpy as jnp

import torchvision.transforms as transforms
import torchvision
import torch
import numpy as np
from torch.distributions.categorical import Categorical

def create_random_perm(key, image_size, n_permutations):
    """
        This function returns a list of array permutation (size of 28*28 = 784) to create permuted MNIST data.
        Note the first permutation is the identity permutation.
        :param n_permutations: number of permutations.
        :return permutations: a list of permutations.
    """
    initial_array = jnp.arange(image_size)
        
    keys = random.split(key, n_permutations - 1)

    def permute(key):
        return random.permutation(key, initial_array)

    permutation_list = vmap(permute)(keys)
    return jnp.vstack([initial_array, permutation_list])

class Permutation(torch.utils.data.Dataset):
    """
    A dataset wrapper that permute the position of features
    """
    def __init__(self, dataset, permute_idx, target_offset):
        super(Permutation,self).__init__()
        self.dataset = dataset
        self.permute_idx = permute_idx
        self.target_offset = target_offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        target = target + self.target_offset
        shape = img.size()
        img = img.view(-1)[self.permute_idx].view(shape)
        return img, target

def _create_task_probs(iters, tasks, task_id, beta=3):
    if beta <= 1:
        peak_start = int((task_id/tasks)*iters)
        peak_end = int(((task_id + 1) / tasks)*iters)
        start = peak_start
        end = peak_end
    else:
        start = max(int(((beta*task_id - 1)*iters)/(beta*tasks)), 0)
        peak_start = int(((beta*task_id + 1)*iters)/(beta*tasks))
        peak_end = int(((beta * task_id + (beta - 1)) * iters) / (beta * tasks))
        end = min(int(((beta * task_id + (beta + 1)) * iters) / (beta * tasks)), iters)

    probs = torch.zeros(iters, dtype=torch.float)
    if task_id == 0:
        probs[start:peak_start].add_(1)
    else:
        probs[start:peak_start] = _get_linear_line(start, peak_start, direction="up")
    probs[peak_start:peak_end].add_(1)
    if task_id == tasks - 1:
        probs[peak_end:end].add_(1)
    else:
        probs[peak_end:end] = _get_linear_line(peak_end, end, direction="down")
    return probs

def _get_linear_line(start, end, direction="up"):
    if direction == "up":
        return torch.FloatTensor([(i - start)/(end-start) for i in range(start, end)])
    return torch.FloatTensor([1 - ((i - start) / (end - start)) for i in range(start, end)])


class ContinuousMultinomialSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, samples_in_batch=128, num_of_batches=69, tasks_samples_indices=None,
                 tasks_probs_over_iterations=None):
        self.data_source = data_source
        assert tasks_samples_indices is not None, "Must provide tasks_samples_indices - a list of tensors," \
                                                  "each item in the list corrosponds to a task, each item of the " \
                                                  "tensor corrosponds to index of sample of this task"
        self.tasks_samples_indices = tasks_samples_indices
        self.num_of_tasks = len(self.tasks_samples_indices)
        assert tasks_probs_over_iterations is not None, "Must provide tasks_probs_over_iterations - a list of " \
                                                         "probs per iteration"
        assert all([isinstance(probs, torch.Tensor) and len(probs) == self.num_of_tasks for
                    probs in tasks_probs_over_iterations]), "All probs must be tensors of len" \
                                                              + str(self.num_of_tasks) + ", first tensor type is " \
                                                              + str(type(tasks_probs_over_iterations[0])) + ", and " \
                                                              " len is " + str(len(tasks_probs_over_iterations[0]))
        self.tasks_probs_over_iterations = tasks_probs_over_iterations
        self.current_iteration = 0

        self.samples_in_batch = samples_in_batch
        self.num_of_batches = num_of_batches

        # Create the samples_distribution_over_time
        self.samples_distribution_over_time = [[] for _ in range(self.num_of_tasks)]
        self.iter_indices_per_iteration = []

        if not isinstance(self.samples_in_batch, int) or self.samples_in_batch <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.samples_in_batch))

    def generate_iters_indices(self, num_of_iters):
        from_iter = len(self.iter_indices_per_iteration)
        for iter_num in range(from_iter, from_iter+num_of_iters):

            # Get random number of samples per task (according to iteration distribution)
            tsks = Categorical(probs=self.tasks_probs_over_iterations[iter_num]).sample(torch.Size([self.samples_in_batch]))
            # Generate samples indices for iter_num
            iter_indices = torch.zeros(0, dtype=torch.int32)
            for task_idx in range(self.num_of_tasks):
                if self.tasks_probs_over_iterations[iter_num][task_idx] > 0:
                    num_samples_from_task = (tsks == task_idx).sum().item()
                    self.samples_distribution_over_time[task_idx].append(num_samples_from_task)
                    # Randomize indices for each task (to allow creation of random task batch)
                    tasks_inner_permute = np.random.permutation(len(self.tasks_samples_indices[task_idx]))
                    rand_indices_of_task = tasks_inner_permute[:num_samples_from_task]
                    iter_indices = torch.cat([iter_indices, self.tasks_samples_indices[task_idx][rand_indices_of_task]])
                else:
                    self.samples_distribution_over_time[task_idx].append(0)
            self.iter_indices_per_iteration.append(iter_indices.tolist())

    def __iter__(self):
        self.generate_iters_indices(self.num_of_batches)
        self.current_iteration += self.num_of_batches
        return iter([item for sublist in self.iter_indices_per_iteration[self.current_iteration - self.num_of_batches:self.current_iteration] for item in sublist])

    def __len__(self):
        return len(self.samples_in_batch)


class DatasetsLoaders:
    def __init__(self, dataset, batch_size=4, num_workers=4, pin_memory=True, **kwargs):
        self.dataset_name = dataset
        self.valid_loader = None
        self.num_workers = num_workers
        if self.num_workers is None:
            self.num_workers = 4

        self.random_erasing = kwargs.get("random_erasing", False)
        self.reduce_classes = kwargs.get("reduce_classes", None)
        self.permute = kwargs.get("permute", False)
        self.target_offset = kwargs.get("target_offset", 0)

        pin_memory = pin_memory if torch.cuda.is_available() else False
        self.batch_size = batch_size


        if dataset == "CONTPERMUTEDPADDEDMNIST":
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))])

            # Original MNIST
            tasks_datasets = [torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)]
            tasks_samples_indices = [torch.tensor(range(len(tasks_datasets[0])), dtype=torch.int32)]
            total_len = len(tasks_datasets[0])
            test_loaders = [torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='./data', train=False,
                                                                                   download=True, transform=transform),
                                                        batch_size=self.batch_size, shuffle=False,
                                                        num_workers=1, pin_memory=pin_memory)]
            self.num_of_permutations = len(kwargs.get("all_permutation"))
            all_permutation = kwargs.get("all_permutation", None)
            for p_idx in range(self.num_of_permutations):
                # Create permuation
                permutation = all_permutation[p_idx]

                # Add train set:
                tasks_datasets.append(Permutation(torchvision.datasets.MNIST(root='./data', train=True,
                                                                             download=True, transform=transform),
                                                  permutation, target_offset=0))

                tasks_samples_indices.append(torch.tensor(range(total_len,
                                                                total_len + len(tasks_datasets[-1])
                                                                ), dtype=torch.int32))
                total_len += len(tasks_datasets[-1])
                # Add test set:
                test_set = Permutation(torchvision.datasets.MNIST(root='./data', train=False,
                                                                  download=True, transform=transform),
                                       permutation, self.target_offset)
                test_loaders.append(torch.utils.data.DataLoader(test_set, batch_size=self.batch_size,
                                                                shuffle=False, num_workers=1,
                                                                pin_memory=pin_memory))
            self.test_loader = test_loaders
            # Concat datasets
            total_iters = kwargs.get("total_iters", None)

            assert total_iters is not None
            beta = kwargs.get("contpermuted_beta", 3)
            all_datasets = torch.utils.data.ConcatDataset(tasks_datasets)

            # Create probabilities of tasks over iterations
            self.tasks_probs_over_iterations = [_create_task_probs(total_iters, self.num_of_permutations+1, task_id,
                                                                    beta=beta) for task_id in
                                                 range(self.num_of_permutations+1)]
            normalize_probs = torch.zeros_like(self.tasks_probs_over_iterations[0])
            for probs in self.tasks_probs_over_iterations:
                normalize_probs.add_(probs)
            for probs in self.tasks_probs_over_iterations:
                probs.div_(normalize_probs)
            self.tasks_probs_over_iterations = torch.cat(self.tasks_probs_over_iterations).view(-1, self.tasks_probs_over_iterations[0].shape[0])
            tasks_probs_over_iterations_lst = []
            for col in range(self.tasks_probs_over_iterations.shape[1]):
                tasks_probs_over_iterations_lst.append(self.tasks_probs_over_iterations[:, col])
            self.tasks_probs_over_iterations = tasks_probs_over_iterations_lst

            train_sampler = ContinuousMultinomialSampler(data_source=all_datasets, samples_in_batch=self.batch_size,
                                                         tasks_samples_indices=tasks_samples_indices,
                                                         tasks_probs_over_iterations=
                                                             self.tasks_probs_over_iterations,
                                                         num_of_batches=kwargs.get("iterations_per_virtual_epc", 1))
            self.train_loader = torch.utils.data.DataLoader(all_datasets, batch_size=self.batch_size,
                                                            num_workers=1, sampler=train_sampler, pin_memory=pin_memory)


        
def ds_padded_cont_permuted_mnist(**kwargs):
    """
    Continuous Permuted MNIST dataset, padded to 32x32.
    :param num_epochs: Number of epochs for the training (since it builds distribution over iterations,
                            it needs this information in advance)
    :param iterations_per_virtual_epc: In continuous task-agnostic learning, the notion of epoch does not exists,
                                        since we cannot define 'passing over the whole dataset'. Therefore,
                                        we define "iterations_per_virtual_epc" -
                                        how many iterations consist a single epoch.
    :param contpermuted_beta: The proportion in which the tasks overlap. 4 means that 1/4 of a task duration will
                                consist of data from previous/next task. Larger values means less overlapping.
    :param permutations: The permutations which will be used (first task is always the original MNIST).
    :param batch_size: Batch size.
    :param num_workers: Num workers.
    """
    dataset = [DatasetsLoaders("CONTPERMUTEDPADDEDMNIST", batch_size=kwargs.get("batch_size", 128),
                               num_workers=kwargs.get("num_workers", 1),
                               total_iters=(kwargs.get("num_epochs")*kwargs.get("iterations_per_virtual_epc")),
                               contpermuted_beta=kwargs.get("contpermuted_beta"),
                               iterations_per_virtual_epc=kwargs.get("iterations_per_virtual_epc"),
                               all_permutation=kwargs.get("permutations", []))]
    test_loaders = [tloader for ds in dataset for tloader in ds.test_loader]
    train_loaders = [ds.train_loader for ds in dataset]

    return train_loaders, test_loaders
    
def error_fn(train_loader, test_loader, epochs):

    for task in range(len(test_loader)):
        for epoch in range(1, epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader[0]):
                continue

            for data, target in test_loader[task]:
                continue

        for i in range(task + 1):
            for data, target in test_loader[i]:
                continue


if __name__ == '__main__':    
    key = random.PRNGKey(0)
    perm_key, key = random.split(key)

    image_size = 784
    n_permutations = 10

    batch_size = 128
    epochs = 20
    alpha = 0.6
    tasks = 10
    iterations_per_virtual_epc = 468

    permutations = create_random_perm(perm_key, image_size, n_permutations)
    permutations = permutations[1:11]
    train_loaders, test_loaders = ds_padded_cont_permuted_mnist(num_epochs=int(epochs * tasks),
                                                                   iterations_per_virtual_epc=iterations_per_virtual_epc,
                                                                   contpermuted_beta=4, permutations=permutations,
                                                                   batch_size=batch_size)

    error_fn(train_loaders, test_loaders, epochs)