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
    

# TODO: Extend the function to support CIFAR-100 and ImageNet datasets.
def prepare_data(double_precision=False, dataset='cifar10'):
    """
    Prepare dataset.
    Returns:
        images(torch.Tensor): Flattened images in the CIFAR-10 dataset.
        all_labels(torch.Tensor): Labels of the images in the CIFAR-10 dataset.
        dataset(str): Name of the dataset. Default is 'cifar10'. Possible values are 'cifar10', 'cifar100', 'tiny-imagenet' and 'adult'.
    """


    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 1024

    if dataset not in ['cifar10', 'cifar100', 'adult', 'tiny-imagenet']:
        raise NotImplementedError("Dataset not supported.")
    
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        n_classes = 10
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform)
        n_classes = 100
    elif dataset == 'adult':
        if os.path.exists('./data/adult'):
            adult_set = AdultDataset(cache_dir='./data/adult', download=False, mode='train', seed=42)
        else:
            adult_set = AdultDataset(cache_dir='./data/adult', download=True, mode='train', seed=42)
        
        if double_precision:
            features_tensor = torch.tensor(adult_set.features, dtype=torch.double)
            labels_tensor = torch.tensor(adult_set.targets, dtype=torch.long)
        else:
            features_tensor = torch.tensor(adult_set.features, dtype=torch.float)
            labels_tensor = torch.tensor(adult_set.labels, dtype=torch.int)
        n_classes = 1
        return features_tensor, labels_tensor
    
    elif dataset == 'tiny-imagenet':
        dataset = TinyImageNetDataset(root='./data/tiny-imagenet-200', transform=transform)
        trainset = dataset.get_train_dataset()
        n_classes = 200

        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        if i == 0:
            all_labels = labels
            images = torch.flatten(inputs, start_dim=1)
        else:
            all_labels = torch.cat((all_labels, labels))
            images = torch.cat((images, torch.flatten(inputs, start_dim=1)))

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


def parse_args():
    """
    Parse the arguments for the script.
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_classes',
                        type=int, 
                        default=10,
                        help='Number of classes in the dataset.')
    
    parser.add_argument("--n_samples",
                        type=int,
                        help="Number of samples to use for the search.")
    
    parser.add_argument("--device",
                        type=str,
                        default='cpu',
                        help="Device to use for the computation.")
    
    parser.add_argument("--display",
                        action='store_true',
                        help="Display the images.",
                        default=False)
    
    parser.add_argument("--dataset",
                        type=str,
                        default='cifar10',
                        help="Dataset to use for the experiment. Possible values are 'cifar10', 'tiny-imagenet' and 'adult'.")
    
    parser.add_argument("--hidden_layers",
                        nargs='+' ,
                        help="Hidden layers structure of the neural network.")
    
    parser.add_argument("--class_bias", 
                        type=float,
                        default=1e12,
                        help="Classification bias value")
    
    parser.add_argument("--weight_scale",
                        type=float,
                        default=1,
                        help="Scale factor for the weight initialization.")
    
    
    return parser.parse_args()