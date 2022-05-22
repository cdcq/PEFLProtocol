import os
from PIL import Image

import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.io import read_image
from torch.utils.data import Subset, Dataset, DataLoader, random_split, sampler


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

def transform_invert(img, transform_train=trans_mnist):
    """
    将data 进行反transfrom操作,并保存图像
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img.dtype, device=img.device)
        std = torch.tensor(norm_transform[0].std, dtype=img.dtype, device=img.device)
        new_img = img.mul(std[:, None, None]).add(mean[:, None, None])

    # img = img.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    # img = np.array(img) * 255
    #
    # if img.shape[2] == 3:
    #     img = Image.fromarray(img.astype('uint8')).convert('RGB')
    # elif img.shape[2] == 1:
    #     img = Image.fromarray(img.astype('uint8').squeeze())
    # else:
    #     raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img.shape[2]))

    return new_img


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

# def get_train_dataset(dataset='mnist', iid=True):
#     edge_dataset = None
#     if dataset == 'mnist':
#         train_dataset = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
#
#     elif dataset == 'cifar':
#         train_dataset = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10)
#
#     elif dataset == "CNNDetection":
#         train_dataset = CNNDection(img_dir=os.path.join("data", "CNNDetection"))
#
#     if iid:
#         num_items = int(len(train_dataset) * 0.3)
#         idxs = iid_sampling(train_dataset, num_items)
#         edge_dataset = Subset(train_dataset, list(idxs))
#         return edge_dataset

def poison_test_idx(test_dataset ,poison_label_swap=1) -> [int]:
    # delete the test data with target label
    leaved_idxs = []
    for idx, (image, label) in enumerate(test_dataset):
        if label != poison_label_swap:
            leaved_idxs.append(idx)

    return leaved_idxs


class DatasetSource:
    def __init__(self, dataset_name, poison_label_swap=1):
        if dataset_name == "mnist":
            self.train_dataset = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
            self.test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=trans_mnist)
        elif dataset_name == "cifar-10":
            self.train_dataset = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10)
            self.test_dataset = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10)
        elif dataset_name == "CNNDetection":
            dataset = CNNDection(img_dir=os.path.join("data", "CNNDetection"))
            train_size, test_size = int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)
            self.train_dataset, self.test_dataset = random_split(dataset, (train_size, test_size)   )

        self.poison_test_idxs = poison_test_idx(test_dataset=self.test_dataset, poison_label_swap=poison_label_swap)
        self.poison_test_dataset = Subset(self.test_dataset, self.poison_test_idxs)

    def get_train_dataloader(self, batch_size=32, frac=0.3, iid=True):
        if iid:
            num_items = int(len(self.train_dataset) * frac)
            idxs = iid_sampling(self.train_dataset, num_items)
            edge_dataset = Subset(self.train_dataset, list(idxs))
            return DataLoader(edge_dataset, batch_size=batch_size, shuffle=True)

    def get_test_dataloader(self, batch_size=32):
        return DataLoader(self.test_dataset, batch_size=32, shuffle=True)

    def get_test_poison_loader(self, batch_size=32):
        return DataLoader(self.poison_test_dataset, batch_size=32, shuffle=True)



# def get_test_dataset(dataset="mnist"):
#     if dataset == "mnist":
#         test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=trans_mnist)
#
#     elif dataset == 'cifar':
#         test_dataset = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10)
#
#     elif dataset == "CNNDetection":
#         test_dataset = CNNDection(img_dir=os.path.join("data", "CNNDetection"))
#
#     num_items = int(len(test_dataset) * 0.4)
#     idxs = iid_sampling(test_dataset, num_items)
#     edge_dataset = Subset(test_dataset, list(idxs))
#     return edge_dataset


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
