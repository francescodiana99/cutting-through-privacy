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
from utils_search import *


def check_span_artificial(observation, images_reconstructed, observation_points, orth_subspace):

    """
    Check if the observation is in the span of the images_reconstructed by projecting all the components"""
    observation = observation.double()
    orth_subspace = orth_subspace.double()
    residual = project_onto_orth_subspace(observation, orth_subspace)

    # for j in range(orth_subspace.shape[0]):
    #     print(f"Dot product with the orthogonal component {j}: {torch.dot(residual, orth_subspace[j])}")

    norm = torch.dot(residual, residual)
    if norm > 1e-5:
        flag = False

    else:
        flag = True

    return flag, residual, norm


def get_activations(acts):
    """ 
    Returns a hook function to store the activations of the first layers' neurons of of a model.
    Args:
        acts(dict): Dictionary to store the activations.
    Returns:
        forward_hook: Function to store the activations."""
    def forward_hook(module, input, output):
        for i in range(output.shape[0]):
            acts[i] = output[i]
    return forward_hook


def check_real_weights(images, labels, model, direction, scale_factor=1, debug=False, display_weights=False):
    """Get the weight distribution associated to the images in the observation"""
    weights = []
    dL_db_list = []
    dL_dA_list = []

    if debug: 
        loss_list = []
        pred_list = []

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for i in range(images.shape[0]):
        optimizer.zero_grad()
        pred = model(images[i])/scale_factor

        if debug: 
            softmax = nn.Softmax(dim=0)
            probs = softmax(pred)
            pred_list.append(probs)
            loss_list.append(loss.item())

        loss = criterion(pred, labels[i])
        loss.backward()
        dL_db = model.fc1.bias.grad[direction].detach().clone()
        dL_dA = model.fc1.weight.grad[direction].detach().clone()

        dL_db_list.append(dL_db)
        dL_dA_list.append(dL_dA)
        weights.append((dL_db.item()))

    sum_dL_dB = sum(dL_db_list)
    dL_dA_tensor = torch.stack(dL_dA_list)
    obs_rec = torch.sum(dL_dA_tensor, dim=0)/sum_dL_dB

    weights_activated = [(i, w) for i, w in enumerate(weights) if w != 0]  
    weights_scaled = [(weights_activated[i][0], (weights_activated[i][1] / sum_dL_dB).item()) for i in range(len(weights_activated))]

    if display_weights:
        print(f"Weights scaled: {weights_scaled}")

    if debug:
        print("-------INSIDE CHECK_REAL_WEIGHTS-------")
        print(f"Weights not scaled: {weights_activated}")  
        print(f"dL_dA from check_real_weights: {torch.sum(dL_dA_tensor, dim=0)/images.shape[0]}")   
        print(f"dL_db from check_real_weights: {sum_dL_dB/images.shape[0]}")   
        print(f"Checking get_observation inside check_real_weights:")
        obs_method = get_observation(images, labels, model, criterion, optimizer, model.fc1.bias, direction, debug=False)
        print(obs_method)
        return  weights_scaled, weights_activated, loss_list, obs_rec

    return weights_scaled, obs_rec


def show_true_images(images):
    """
    Display the images in the dataset.
    Args:
        images(torch.Tensor): Flattened images tensor.
    """
    for i in range(images.shape[0]):
        std = torch.tensor([0.5, 0.5, 0.5])
        mean = torch.tensor([0.5, 0.5, 0.5])

        image = images[i].view(3, 32, 32)
        image = image.permute(1, 2, 0)
        image = image * std + mean
        plt.imshow(image)
        plt.show()


def get_image_gradients(images, labels, model, criterion, optimizer, b, direction):
    """
    Get gradients values for each image in the batch
    """
    model.fc1.bias.data = b.detach().clone()
    model.train()
    da_dL_list = []
    db_dL_list = []
    db_dL_large_list = []
    db_dL_class_list = []
    db_dA_class_list = []

    for i in range(images.shape[0]):
        optimizer.zero_grad()

        preds = model(images[i])
        loss = criterion(preds, labels[i])

        loss.backward()
        if model.fc1.bias.grad.data[direction] != 0:
            da_dL_list.append(model.fc1.weight.grad.data[direction].detach().clone())
            db_dL_list.append(model.fc1.bias.grad.data[direction].detach().clone())
            db_dL_large_list.append(model.fc1.bias.grad.data[-1].detach().clone())
            db_dA_class_list.append(model.fc2.weight.grad.data[direction].detach().clone())
            db_dL_class_list.append(model.fc2.bias.grad.data[direction].detach().clone())
    
    return da_dL_list, db_dL_list, db_dL_large_list


