import os

import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.io import read_image
from torch.utils.data import Subset, Dataset, DataLoader, random_split, sampler

Transform_CUS = {
    0: transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),

    1: transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),

    2: transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
}

def transform_invert(img, task):
    """
    将data 进行反transfrom操作,并保存图像
    :param img: tensor
    :param task: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(Transform_CUS[task]):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), Transform_CUS[task].transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img.dtype, device=img.device)
        std = torch.tensor(norm_transform[0].std, dtype=img.dtype, device=img.device)
        new_img = img.mul(std[:, None, None]).add(mean[:, None, None])

    return new_img


class CNNDection(Dataset):
    def __init__(self, img_dir, transform=Transform_CUS[2], target_transform=None):
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
            self.train_dataset = datasets.MNIST('data/mnist/', train=True, download=True, transform=Transform_CUS[0])
            self.test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=Transform_CUS[0])
        elif dataset_name == "cifar-10":
            self.train_dataset = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=Transform_CUS[1])
            self.test_dataset = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=Transform_CUS[1])
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


def iid_sampling(dataset, num_items):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_items:
    :return: dict of image index
    """
    all_idxs = [i for i in range(len(dataset))]
    return set(np.random.choice(all_idxs, num_items, replace=True))
