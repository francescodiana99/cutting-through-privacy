import logging
import torch
import torchvision
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

import matplotlib.pyplot as plt

import bisect
import random
import math
import argparse
import os
import time

from datasets.adult.adult import AdultDataset
from datasets.tiny_imagenet.tiny_imagenet import TinyImageNetDataset


class NN(nn.Module):
    """
    Simple fully connected neural network with one hidden layer.
    """
    def __init__(self, input_dim, n_classes=100, hidden_dim=1000, weight_scale=1):
        """
        Args:
            input_dim(int): Dimension of the input.
            n_classes(int): Number of classes in the dataset.
            hidden_dim(int): Dimension of the hidden layer.
        """
        super(NN, self).__init__()
        self.hidden_dim = hidden_dim
        if hidden_dim != 0:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, n_classes)
        else:
            self.fc1 = nn.Linear(input_dim, n_classes)

        torch.nn.init.uniform_(self.fc1.weight, 0, 1*weight_scale)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        if self.hidden_dim != 0:
            x = self.fc2(x)
        return x
    

def prepare_data(double_precision=False, dataset='cifar10', data_dir='./data/', n_samples=32, model_type='fc'):
    """
    Prepare dataset.
    Args:
        double_precision(bool): Whether to use double precision for the computations. Default is False.
        dataset(str): Name of the dataset to use. Default is 'cifar10'. 
        Available options are 'cifar10', 'cifar100', 'adult', 'tiny-imagenet' and 'imagenet'.
        data_dir(str): Directory to store the data. Default is './data/'.
        n_samples(int): Number of samples to use from the dataset. Default is 32.
        model_type(str): Type of the model to use. Default is 'fc'. Available options are 'fc' and 'cnn'.
    Returns:
        images(torch.Tensor): Flattened images in the CIFAR-10 dataset.
        all_labels(torch.Tensor): Labels of the images in the CIFAR-10 dataset.
        n_classes(str): Number of classes in the dataset.
    """

    rnd_g = torch.Generator()
    if dataset not in ['cifar10', 'cifar100', 'adult', 'tiny-imagenet', 'imagenet']:
        raise NotImplementedError("Dataset not supported.")
    
    if dataset == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor()
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    else:
        transform = transforms.Compose(
            [transforms.ToTensor()])
            #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                download=True, transform=transform)
        n_classes = 10
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                                download=True, transform=transform)
        n_classes = 100
    elif dataset == 'adult':
        if os.path.exists(data_dir):
            adult_set = AdultDataset(cache_dir=data_dir, download=False, mode='train', seed=42)
        else:
            adult_set = AdultDataset(cache_dir=data_dir, download=True, mode='train', seed=42)
        
        if double_precision:
            features_tensor = torch.tensor(adult_set.features, dtype=torch.double)
            labels_tensor = torch.tensor(adult_set.targets, dtype=torch.long)
        else:
            features_tensor = torch.tensor(adult_set.features, dtype=torch.float)
            labels_tensor = torch.tensor(adult_set.labels, dtype=torch.int)
        n_classes = 1
        return features_tensor, labels_tensor, n_classes
    
    elif dataset == 'tiny-imagenet':
        dataset = TinyImageNetDataset(root=data_dir, transform=transform)
        trainset = dataset.get_train_dataset()
        n_classes = 200
    
    elif dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(root=data_dir, split='val', transform=transform)
        trainset = dataset
        n_classes = 1000

    random_indices= torch.randperm(len(trainset), generator=rnd_g)[:n_samples]
    trainset = torch.utils.data.Subset(trainset, random_indices)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=n_samples,
                                            shuffle=True, num_workers=2)
    if model_type == 'fc':
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if i == 0:
                all_labels = labels
                images = torch.flatten(inputs, start_dim=1)
            else:
                all_labels = torch.cat((all_labels, labels))
                images = torch.cat((images, torch.flatten(inputs, start_dim=1)))
    else:
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if i == 0:
                all_labels = labels
                images = inputs
            else:
                all_labels = torch.cat((all_labels, labels))
                images = torch.cat((images, inputs))
                
    if double_precision:
                images = images.double()
                labels = labels.double()
    return images, all_labels, n_classes


def set_seeds(seed):
    """
    Set the random seed for reproducibility.

    Parameters:
    - seed (int): The random seed to be set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)    

def configure_logging():
    """
    Set up logging based on verbosity level
    """
    # TODO: this should be fixed. Opacus changes the default level to WARNING
    logging.basicConfig(level=logging.INFO)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    root_logger.setLevel(logging.INFO)

