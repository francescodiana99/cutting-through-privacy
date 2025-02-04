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

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


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

def get_class_inputs(input_dict, i):
    def forward_hook(module, inputs, outputs):
        if input_dict.get(i) is None:
            input_dict[i] = [inputs[0]]
        else:
            input_dict[i].append(inputs[0])
    return forward_hook

def get_conv_res(input_dict, output_dict, i):
    def forward_hook(module, inputs, outputs):
        if input_dict.get(i) is None:
            input_dict[i] = [inputs[0]]
        if output_dict.get(i) is None:
            output_dict[i] = [outputs[0]]
        else:
            input_dict[i].append(inputs[0])
    return forward_hook

def capture_gradient(module, grad_input, grad_output):
    print("Gradient of x_out:", grad_output[0])

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
        loss = criterion(pred, labels[i])

        if debug: 
            softmax = nn.Softmax(dim=0)
            probs = softmax(pred)
            pred_list.append(probs)
            loss_list.append(loss.item())

        loss.backward()
        dL_db = model.layers[0].bias.grad[direction].detach().clone()
        # z_1 = model.fc1.weight @ images[i] + model.fc1.bias
        # z_2 = model.fc2.weight @ z_1 + model.fc2.bias
        # z_2_no_label = torch.cat((z_2[:labels[i]], z_2[labels[i]+1:]), dim=0)
        # softmax_z_2_no_label = torch.cat((softmax(z_2)[:labels[i]], softmax(z_2)[labels[i]+1:]), dim=0)
        # w_2_no_label = torch.cat((model.fc2.weight[:labels[i]], model.fc2.weight[labels[i]+1:]), dim=0)
        # dL_db_manual = model.fc2.weight[labels[i], direction] * (-1 + torch.exp(z_2[labels[i]])/torch.sum(torch.exp(z_2))) \
        #     + (torch.exp(z_2_no_label) / torch.sum(torch.exp(z_2)) @  w_2_no_label[:, direction])
        # dL_db_manual_with_softmax = model.fc2.weight[labels[i], direction] * (-1 + softmax(z_2)[labels[i]]) \
        #     + softmax_z_2_no_label @  w_2_no_label[:, direction]
        
        # if debug:
        #     if dL_db.item() != 0:
        #         print(f"Difference between dL_db and dL_db_manual: {dL_db - dL_db_manual_with_softmax}")
        dL_dA = model.layers[0].weight.grad[direction].detach().clone()

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
        print(f" weifhts scaled: {weights_scaled}")
        # print(f"dL_dA from check_real_weights: {torch.sum(dL_dA_tensor, dim=0)/images.shape[0]}")   
        # print(f"dL_db from check_real_weights: {sum_dL_dB/images.shape[0]}")   
        # print(f"Checking get_observation inside check_real_weights:")
        # obs_method = get_observation(images, labels, model, criterion, optimizer, model.fc1.bias, direction, debug=False)
        # print(obs_method)
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


def get_ssim(img_1, img_2, dataset_name):
    """
    Restore the original shape of img_1 and img_2 and compute the SSIM between them."""
    if dataset_name in ['cifar10', 'cifar100']:
        H, W, C = 32, 32, 3
    elif dataset_name == 'tiny-imagenet':
        H, W, C = 64, 64, 3
    elif dataset_name == 'imagenet':
        H, W, C = 224, 224, 3
    else:
        raise NotImplementedError("Dataset not supported.")

    img_1 = img_1.cpu().reshape(H, W, C).numpy()
    img_2 = img_2.cpu().reshape(H, W, C).numpy()

    ssim_metric = ssim(img_1, img_2, channel_axis=2, data_range=1) 
    return ssim_metric


def get_psnr(input_1, input_2, data_range=1):
    """
    Compute the PSNR between two inputs.
    """
    input_1 = input_1.cpu().numpy()
    input_2 = input_2.cpu().numpy()
    psnr_metric = psnr(input_1, input_2, data_range=data_range) 

    return psnr_metric


# def couple_inputs(rec_inputs_list, true_inputs_list, dataset_name='cifar10'):
#     """
#     Find the correspondencies between images, according to SSIM if data are images, otherwise according to the maximum norm difference."""
#     couples = []
#     if dataset_name in ['cifar10', 'cifar100', 'tiny-imagenet', 'imagenet']:
#         for i in range(len(rec_inputs_list)):
#             img_1 = rec_inputs_list[i]
#             ssim_list = [get_ssim(img_1, img_2, dataset_name) for img_2 in true_inputs_list]
#             max_ssim = max(ssim_list)
#             idx = ssim_list.index(max_ssim)
#             couples.append((rec_inputs_list[i], true_inputs_list[idx], max_ssim))
#     else:
#         for i in range(len(rec_inputs_list)):
#             img_1 = rec_inputs_list[i]
#             diff_list = [torch.norm(img_1 - img_2).item() for img_2 in true_inputs_list]
#             min_diff = min(diff_list)
#             idx = diff_list.index(min_diff)
#             couples.append((rec_inputs_list[i], true_inputs_list[idx], min_diff))
#     return couples


def couple_inputs(rec_inputs_list, true_inputs_list, dataset_name='cifar10', device='cuda'):
    """
    Find the correspondencies between images, according to SSIM if data are images, otherwise according to the maximum norm difference."""

    rec_inputs = torch.stack(rec_inputs_list)
    true_inputs = torch.stack(true_inputs_list)

    N = rec_inputs.shape[0]
    # M = true_inputs.shape[0]

    couples = []
    batch_size = 10
    for i in range(0, N, 10):
        if i + batch_size > N:
            batch_size = N - i
        batch = rec_inputs[i:i + batch_size]
        

        diff = batch.unsqueeze(1) - true_inputs.unsqueeze(0)  # Shape: [batch_size, M, D]
        norm_diff = torch.norm(diff, dim=2)  # Shape: [batch_size, M]

        # Find the best match for each recovered image in the batch
        min_diff, idxs = torch.min(norm_diff, dim=1)  # min_diff: [batch_size], idxs: [batch_size]\

        # Append results to the final list
        for j in range(batch.shape[0]):
            couples.append((rec_inputs_list[i + j], true_inputs_list[idxs[j]], min_diff[j].item()))
    
    return couples

        


def max_norm_difference(tensor):
    """
    Compute the maximum norm difference between the first row and all other rows.
    
    Args:
        tensor (torch.Tensor): The input tensor (2D).
        
    Returns:
        max_diff (float): The maximum norm difference.
        position (int): The index of the row with the maximum norm difference.
    """
    first_row = tensor[0]
    max_diff = 0
    position = -1

    for i, row in enumerate(tensor):
        norm_diff = torch.norm(row - first_row).item()
        if norm_diff > max_diff:
            max_diff = norm_diff
            position = i

    return max_diff, position