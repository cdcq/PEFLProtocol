import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])


def get_train_dataset(dataset='mnist', iid=True):
    if dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        if iid:
            num_items = int(len(dataset_train) * 0.4)
            idxs = iid_sampling(dataset_train, num_items)
            edge_dataset = Subset(dataset_train, list(idxs))
            return edge_dataset

    if dataset == "posioned_mnist":
        # TODO: 董
        return


def get_test_dataset(dataset="mnist"):
    if dataset == "mnist":
        test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=trans_mnist)
        return test_dataset


def iid_sampling(dataset, num_items):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_items:
    :return: dict of image index
    """
    all_idxs = [i for i in range(len(dataset))]
    return set(np.random.choice(all_idxs, num_items, replace=True))
