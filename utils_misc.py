import json
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
from datasets.harus.harus import HARUSDataset
from datasets.tiny_imagenet.tiny_imagenet import TinyImageNetDataset
    

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
    if dataset not in ['cifar10', 'cifar100', 'adult', 'harus', 'tiny-imagenet', 'imagenet']:
        raise NotImplementedError("Dataset not supported.")
    
    if dataset == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
            labels_tensor = torch.tensor(adult_set.targets, dtype=torch.long)

        sample_idx = torch.randperm(len(labels_tensor), generator=rnd_g)[:n_samples]

        features_tensor = features_tensor[sample_idx]
        labels_tensor = labels_tensor[sample_idx]
        n_classes = 1
    
        return features_tensor, labels_tensor, n_classes
    
    elif dataset == 'harus':
        if os.path.exists(data_dir):
            harus_set = HARUSDataset(cache_dir=data_dir, download=False, mode='train')
        else:
            harus_set = HARUSDataset(cache_dir=data_dir, download=True, mode='train')

        if double_precision:
            features_tensor = torch.tensor(harus_set.get_features(), dtype=torch.double)
            labels_tensor = torch.tensor(harus_set.get_targets(), dtype=torch.long)
        else:
            features_tensor = torch.tensor(harus_set.get_features(), dtype=torch.float)
            labels_tensor = torch.tensor(harus_set.get_targets(), dtype=torch.long)

        sample_idx = torch.randperm(len(labels_tensor), generator=rnd_g)[:n_samples]

        features_tensor = features_tensor[sample_idx]
        labels_tensor = labels_tensor[sample_idx]
        n_classes = 6

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
    logging.basicConfig(level=logging.INFO)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    root_logger.setLevel(logging.INFO)


def save_results(data_path, result_dict, com_round, rec_inputs=None):
    """
    Save the results to a file.

    Parameters:
    - data_path (str): The path to the file where the results will be saved.
    - result_dict (dict): The dictionary containing the results.
    - com_round (int): The communication round  number.
    - rec_inputs (torch.Tensor): Tensor containing the reconstructed input. Default is None.
    """

    os.makedirs(data_path, exist_ok=True)

    file_path = os.path.join(data_path, f"results.json")

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            results = json.load(f)
        results[f"{com_round}"] = result_dict
    else:
        results = {f"{com_round}": result_dict}
    with open(file_path, 'w') as f:
        json.dump(results, f)
    logging.info(f"Results saved in {file_path}.")

    if rec_inputs is not None:
        torch.save(rec_inputs, os.path.join(data_path, f"{com_round}.pt"))


def restore_images(images, device='cpu', display=False, title=None, dataset_name='cifar10'):
    if dataset_name != 'imagenet':
        std = torch.tensor([0.5, 0.5, 0.5]).to(device)
        mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
    else:
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    images_scaled = []
    for image in images:
        image = image.to(device)
        if dataset_name in ['cifar10', 'cifar100']:
            image = image.view(3, 32, 32)
        elif dataset_name == 'tiny-imagenet':
            image = image.view(3, 64, 64)
        elif dataset_name == 'imagenet':
            image = image.view(3, 224, 224)
        else:
            raise ValueError("Dataset not supported")
        image = image.permute(1, 2, 0)
        image = image * std + mean
        images_scaled.append(image)

    if display:

        fig, axs = plt.subplots(1, len(images_scaled))

        for i in range(len(images_scaled)):
            axs[i].imshow((images_scaled[i].to('cpu').numpy()))
            axs[i].axis('off')
        if title is not None:
            plt.title(title)
        plt.show()
    
    return image

