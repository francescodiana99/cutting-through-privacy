import torch
import torchvision
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

import bisect
import random
import math
import argparse
import os

from torchvision.datasets import ImageFolder

class NN(nn.Module):
    def __init__(self, input_dim, n_classes=100, hidden_dim=1000 ):
        super(NN, self).__init__()
        self.hidden_dim = hidden_dim
        if hidden_dim != 0:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, n_classes)
        else:
            self.fc1 = nn.Linear(input_dim, n_classes)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        if self.hidden_dim != 0:
            x = self.fc2(x)
        return x
    
class TinyImageNetDataset:
    """
    Dataset Class for Tiny Imagenet dataset

    Args:
        root(str): Path to the root folder of the Tiny ImageNet dataset. 
        def
    """
    def __init__(self, root, transform):
        self.root = os.path.expanduser(root)   
        self.transform = transform
        
        self.train_dataset = self._get_train_dataset(self.transform)
        self.val_dataset = self._get_val_dataset(self.transform)
    
    def _get_train_dataset(self, transform):
        train_dir = os.path.join(self.root)
        return ImageFolder(train_dir, transform=transform) 
    

    def _get_val_dataset(self, transform):
        val_dir = os.path.join(self.root)
        return ImageFolder(val_dir, transform=transform)
    
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_val_dataset(self):
        return self.val_dataset
    

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        help='Dataset to use',
        choices=['cifar10', 'cifar100', 'tiny'])
    
    return parser.parse_args()

def get_activations(acts):
    def forward_hook(module, input, output):
        acts['first_neuron'] = output
    return forward_hook
    
def get_weigths_distribution(images, labels, n_classes, hidden_dim, isolated_samples=2):
    """Return the weight distribution associated to x1 and x2"""
    image_size = images.shape[1]
    acts = {}

    model = NN(image_size, n_classes=n_classes, hidden_dim=hidden_dim)
    b_list = []
    for x in images:
        b = -torch.matmul(model.fc1.weight[0], x)
        b_list.append(b)
    b_list.sort()
    model.fc1.bias.data[0] = b
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()

    optimizer.zero_grad
    # register hook to keep track of the forward propagation
    h = model.fc1.register_forward_hook(get_activations(acts))
    preds = model(images)
    loss = criterion(preds, labels)
    loss.backward()
    dL_dA = model.fc1.weight.grad[0]
    dL_db = model.fc1.bias.grad[0]
    x_c = dL_dA / dL_db
    # check the number of activations
    act_mask = (acts['first_neuron'][:, 0] > 0)
    h.remove()
    n_act = torch.sum((acts['first_neuron'][:, 0] > 0).int())
    print(f"Number of activations: {n_act}")

    act_imgs = images[act_mask]
    act_labels = labels[act_mask]
    weights, struct_sim , same_class = get_weights(act_imgs, act_labels, model)
    print(" Weights: ", weights)
    return weights, struct_sim, same_class


def get_weights(images, labels, model):
    """Return the weight distribution associated to x1 and x2"""
    weights = []
    dL_db_list = []
    dL_dA_list = []
    loss_list = []
    rec_img_list = []

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for i in range(images.shape[0]):
        optimizer.zero_grad()
        pred = model(images[i])
        loss = criterion(pred, labels[i])
        loss_list.append(loss)
        loss.backward()
        dL_db = model.fc1.bias.grad[0]
        dL_dA = model.fc1.weight.grad[0]

        dL_db_list.append(dL_db)
        dL_dA_list.append(dL_dA)

        x_rec = dL_dA / dL_db
        rec_img_list.append(x_rec)

        weights.append((dL_db.item()))
    if labels[0] == labels[1]:
        same_class = True
    else:
        same_class = False
        mean = torch.tensor([0.5, 0.5, 0.5,])
        std = torch.tensor([0.5, 0.5, 0.5])
        # x_rec_to_disp = []
        # for x_rec in rec_img_list:
            
        #     x_rec_disp = x_rec.view(3, 32, 32)
        #     x_rec_disp = x_rec_disp.permute(1, 2, 0)
        #     x_rec_disp = x_rec_disp * std + mean
        #     x_rec_to_disp.append(x_rec_disp)

        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(x_rec_to_disp[0])
        # axs[0].axis('off')
        # axs[1].imshow(x_rec_to_disp[1])
        # axs[1].axis('off')
        # plt.show()

    # struct_sim = ssim(rec_img_list[0].numpy(), rec_img_list[1].numpy(), data_range=2)
    sum_dL_dB = sum(dL_db_list)
    print("Weights not scaled: ", weights)
    weights_scaled = [weights[i] / sum_dL_dB for i in range(len(weights))]
    
    sum_dL_dA = sum(dL_dA_list)
        
    # print(f"Sanity check")
    # print(f"Observed x_c: {x_c}")
    # print(f"Computed x_c: {computed_x_c}")


    return weights_scaled, None, same_class

    
def main():

    args = parse_args()

    # download and preprocess data
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 512

    if args.dataset == 'cifar10':

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        n_classes = 10
        
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        n_classes = 100

    elif args.dataset == 'tiny':
        dataset = TinyImageNetDataset(root='./data/tiny-imagenet-200', transform=transform)
        trainset = dataset.get_train_dataset()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        n_classes = 200

    
    for i, data in enumerate(trainloader, 0):
        inputs, y = data
        if i == 0:
            images = torch.flatten(inputs, start_dim=1)
            labels = y 
        else:
            images = torch.cat((images, torch.flatten(inputs, start_dim=1)))
            labels = torch.cat((labels, y))
    
    num_samples=4
    weights_list = []
    struct_sim_list = []
    weight_class_list = []

    for trial in range(5000):
        sample = np.random.choice(range(images.shape[0]), size=num_samples, replace=False)
        images_client = images[sample]
        labels_client = labels[sample]
        reconstructed_images = []
        weights, struct_sim, same_class = get_weigths_distribution(images_client, labels_client, n_classes=n_classes, hidden_dim=1024, isolated_samples=4)
        if weights is not None:
            weights_list.append(weights)
            weight_class_list.append((weights, same_class))
            struct_sim_list.append(struct_sim)

    weights_list = np.array(weights_list)
    print(f"number of points: {weights_list.shape[0]}")

    for i in range(weights_list.shape[1]):
        plt.hist(x=weights_list[:,i], bins=50)
        plt.title(f"Distribution of alpha_{i} ({len(weights_list)} trials)")
        plt.xlabel('alpha')
        plt.ylabel('count')
        plt.show()

    zoomed_list = [x for x in weights_list[:,0] if -1 <= x <= 1.]
    print(f"percentage of points in [-1,1]: {len(zoomed_list)/len(weights_list)}")
    plt.hist(x=zoomed_list, bins=50)
    plt.title(f"Distribution of alpha  (only [-1,1] range)")
    plt.xlabel('alpha')
    plt.ylabel('count')
    plt.show()
    


    # for i in range(weights_list.shape[1]):
    #     weights_same_class = [j[0][i] for j in weight_class_list if j[1] == True]
    #     plt.hist(x=weights_same_class, bins=50)
    #     plt.title(f"Distribution of alpha (same class)")
    #     plt.show()   

    #     weights_diff_class = [j[0][i] for j in weight_class_list if j[1] == False]
    #     plt.hist(x=weights_diff_class, bins=50)
    #     plt.title(f"Distribution of alpha  (diff class)")
    #     plt.show()   


    # plot_list = [(weights_list[i,0], struct_sim_list[i]) for i in range(weights_list.shape[0])]
    # plot_list.sort(key=lambda x: x[0])
    # x = [i for i, j in  plot_list]
    # y = [j for i, j in plot_list]
    # plt.scatter(x, y)
    # plt.xlabel('alpha')
    # plt.ylabel('ssim')
    # plt.show()
    

if __name__ == '__main__':
    main()


