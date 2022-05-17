import argparse
import copy
from logging import Logger
import random
import numpy as np
import yaml
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize((0.1307,), (0.3081,))
                        ]))
test_dataset = datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]))
   
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
train_dataset_cifar = dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
def get_train_dataset(dataset='mnist', iid=True):
    edge_dataset = None
    if dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        if iid:
            num_items = int(len(dataset_train) * 0.4)
            idxs = iid_sampling(dataset_train, num_items)
            edge_dataset = Subset(dataset_train, list(idxs))
    elif dataset == 'cifar':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        if iid:
            num_items = int(len(dataset_train) * 0.4)
            idxs = iid_sampling(dataset_train, num_items)
            edge_dataset = Subset(dataset_train, list(idxs))
    return edge_dataset


def get_test_dataset(dataset="mnist"):
    if dataset == "mnist":
        test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=trans_mnist)
    elif dataset == 'cifar':
        #dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        test_dataset = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
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
def get_train_old(all_range, model_no,dataname): #将train_dataset分给参与者
    """
    This method equally splits the dataset.
    :param params:
    :param all_range:
    :param model_no:
    :return:
    """
    if dataname=='mnist':

        data_len = int(len(train_dataset) / 100)
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=64,
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                sub_indices))
    elif dataname=='cifar':
        data_len = int(len(train_dataset_cifar) / 100)
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = torch.utils.data.DataLoader(train_dataset_cifar,
                                            batch_size=64,
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                sub_indices))

    return train_loader

def load_data(datasetnme):
    if datasetnme=='mnist':
        ## sample indices for participants that are equally
        all_range = list(range(len(train_dataset)))
    elif datasetnme=='cifar':
        all_range = list(range(len(train_dataset_cifar)))
    random.shuffle(all_range)
    train_loaders = [(pos, get_train_old(all_range, pos,datasetnme))
                        for pos in range(100)]
    #print('train loaders done')
    #self.train_data = train_loaders
    return(train_loaders)