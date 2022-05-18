import os

import numpy as np
from torchvision import datasets, transforms
from torchvision.io import read_image
from torch.utils.data import Subset, Dataset

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])

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

def get_train_dataset(dataset='mnist', iid=True, sample_frac=0.4):
    if dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)

    if dataset == "posioned_mnist":
        # TODO: 董
        pass

    if dataset == "CNNDetection":
        dataset_train = CNNDection(img_dir=os.path.join("data", "CNNDetection"))

    if iid:
        num_items = int(len(dataset_train) * sample_frac)
        idxs = iid_sampling(dataset_train, num_items)
        edge_dataset = Subset(dataset_train, list(idxs))
        return edge_dataset


def get_test_dataset(dataset="mnist"):
    if dataset == "mnist":
        test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=trans_mnist)

    if dataset == "CNNDetection":
        test_dataset = CNNDection(img_dir=os.path.join("data", "CNNDetection"))

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
