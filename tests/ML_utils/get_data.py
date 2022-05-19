import os
import numpy as np
from torch import nn
from torchvision import datasets, transforms
from torchvision.io import read_image
from torch.utils.data import Subset, Dataset


trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])

trans_cifar10 = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# trans_cifar10 = transforms.Compose([transforms.ToTensor(),
#                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trans_CNNDetction = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class CNNDection(Dataset):
    def __init__(self, img_dir, transform=trans_CNNDetction, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        files = os.listdir(self.img_dir)
        return len(files)

    def __getitem__(self, idx):
        if idx % 2 == 0:
            # idx 为偶数时取real, lable为0
            img_path = os.path.join(self.img_dir, "{:0>4d}_real.png".format(idx // 2))
            label = 0
        else:
            # idx为奇数时取fake, label为1
            img_path = os.path.join(self.img_dir, "{:0>4d}_fake.png".format(idx // 2))
            label = 1
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def get_train_dataset(dataset='mnist', iid=True):
    edge_dataset = None
    if dataset == 'mnist':
        train_dataset = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)

    elif dataset == 'cifar':
        train_dataset = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10)

    elif dataset == "CNNDetection":
        train_dataset = CNNDection(img_dir=os.path.join("data", "CNNDetection"))

    if iid:
        num_items = int(len(train_dataset) * 0.3)
        idxs = iid_sampling(train_dataset, num_items)
        edge_dataset = Subset(train_dataset, list(idxs))
        return edge_dataset


def get_test_dataset(dataset="mnist"):
    if dataset == "mnist":
        test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=trans_mnist)

    elif dataset == 'cifar':
        test_dataset = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10)

    elif dataset == "CNNDetection":
        test_dataset = CNNDection(img_dir=os.path.join("data", "CNNDetection"))

    num_items = int(len(test_dataset) * 0.4)
    idxs = iid_sampling(test_dataset, num_items)
    edge_dataset = Subset(test_dataset, list(idxs))
    return edge_dataset


def iid_sampling(dataset, num_items):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_items:
    :return: dict of image index
    """
    all_idxs = [i for i in range(len(dataset))]
    return set(np.random.choice(all_idxs, num_items, replace=True))


# def get_train_old(all_range, model_no,dataname):
#     """
#     This method equally splits the dataset.
#     :param params:
#     :param all_range:
#     :param model_no:
#     :return:
#     """
#     if dataname=='mnist':
#
#         data_len = int(len(train_dataset) / 100)
#         sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
#         train_loader = torch.utils.data.DataLoader(train_dataset,
#                                             batch_size=64,
#                                             sampler=torch.utils.data.sampler.SubsetRandomSampler(
#                                                 sub_indices))
#     elif dataname=='cifar':
#         data_len = int(len(train_dataset_cifar) / 100)
#         sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
#         train_loader = torch.utils.data.DataLoader(train_dataset_cifar,
#                                             batch_size=64,
#                                             sampler=torch.utils.data.sampler.SubsetRandomSampler(
#                                                 sub_indices))
#
#     return train_loader
#
# def load_data(datasetnme):
#     if datasetnme=='mnist':
#         ## sample indices for participants that are equally
#         all_range = list(range(len(train_dataset)))
#     elif datasetnme=='cifar':
#         all_range = list(range(len(train_dataset_cifar)))
#     random.shuffle(all_range)
#     train_loaders = [(pos, get_train_old(all_range, pos,datasetnme))
#                         for pos in range(100)]
#     return(train_loaders)
