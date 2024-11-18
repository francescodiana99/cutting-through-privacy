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

from utils_misc import *
from utils_test import *


def projection(x, y):
    """
    Project x onto y
    """
    return (torch.dot(x, y)/torch.dot(y, y))*y


def project_onto_orth_subspace(x, orth_subspace):
    """
    Project x onto the the vectors in orth_subspace.
    """

    residual = x.detach().clone()
    orth_subspace = orth_subspace.detach().clone()
    
    for i in range(orth_subspace.shape[0]):
        residual -= projection(x, orth_subspace[i])
    return residual


def check_span(orth_subspace, observation, norm_threshold=1e-5, debug=False):
    """
    Checks if x is in the span of the vectors in orth_subspace.
    Args:
        orth_subspace(torch.Tensor): Orthogonal subspace.
        observation(torch.Tensor): Observation to check.
        debug(bool) (optional): Debug flag. If True, the function returns the norm of the residual after the projection. Default is False.
    Returns:
        bool: True if the observation is in the span of the vectors in the orthogonal subspace.
        float: Norm of the projection
    """

    observation = observation.double()
    orth_subspace = orth_subspace.double()
    residual = project_onto_orth_subspace(observation, orth_subspace)
    norm = torch.dot(residual, residual)
    if debug:
        print(f"Norm of the residual: {torch.dot(residual, residual)} | threshold: {norm_threshold} | Is in span: {norm < norm_threshold}")
    return norm < norm_threshold, norm


def find_corresponding_strips_sequential(strips_dict):
    """
    Find the corresponding strips for each direction, by executing the following steps:
    1. Fix a direction v.
    2. Compute the orthogonal subspace spanned by v.
    3. For each observation u in the direction v, project it onto the orthogonal subspace spanned by v.
    4. If the projection norm is zero, then the observation u is in the span of the orthogonal subspace
    This means that the two observations correspond to the same image.
    5. Add the observation to the corresponding observation list.
    6. Repeat the process for all the directions.
    Args:
        strips_dict(dict): Dictionary containing the observation points for each direction.
    Returns:
        corresponding_strips(dict): Dictionary containing the corresponding strips for each direction.
    """

    v = strips_dict[0] # fix direction v
    v_orth = v[0][1].unsqueeze(0) # first element of the orthogonal supspace spanned by v
    strips_dict.pop(0)

    for i in range(len(v)):
        if i != 0:
            residual = project_onto_orth_subspace(v[i][1], v_orth)
            v_orth = torch.cat((v_orth, residual.unsqueeze(0)), 0)

    corresponding_obs = {i: v[i][1].unsqueeze(0) for i in range(len(v))}
    corresponding_bias = {i: [v[i][0].item()] for i in range(len(v))}
    for d in strips_dict.keys():
        u = strips_dict[d]
        for i in range(v_orth.shape[0]):
            curr_orth = v_orth[:i + 1] # select the first i orthogonal components in v_orth (corresponding to the component at the i-th observation)

            not_found = True
            j = 0

            while not_found:
                # project the observation u[j] onto the orthogonal subspace spanned by the first i orthogonal components of v
                residual = project_onto_orth_subspace(u[j][1], curr_orth)
                # if the residual is zero, then the observation u[j] is in the span of the orthogonal subspace
                if torch.dot(residual, residual) < 1e-5:
                    # add the observation to the corresponding observation list
                    corresponding_obs[i] = torch.cat((corresponding_obs[i], u[j][1].unsqueeze(0)), 0)
                    corresponding_bias[i].append(u[j][0].item())
                    not_found = False
                    corr = u.pop(j)
                    assert torch.equal(corr[1],corresponding_obs[i][-1])


                else:
                    # add the residual to the orthogonal subspace
                    curr_orth = torch.cat((curr_orth, residual.unsqueeze(0)), 0)
                    j += 1
                    if j == len(u):
                        not_found = False
                        raise ValueError(f"Warning: an image correspondence has not been detected even though it should")
                        # no_corr_list.append(v[i])
                        # for i in range(curr_orth.shape[0]):
                        #     print(f"Dot product with the orthogonal component {i}: {torch.dot(residual, curr_orth[i])}")

    return corresponding_obs, corresponding_bias


def restore_images(images, device='cpu', display=False, title=None):

    std = torch.tensor([0.5, 0.5, 0.5]).to(device)
    mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
    images_scaled = []
    for image in images:
        image = image.to(device)
        image = image.view(3, 32, 32)
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


def solve_linear_system(observation, directions, b, images, device='cpu'):
    """
    Solve a linear sistem to find the k-th image.
    Args:
        observation: tensor of the observation
        directions: tensor of the directions
        b: list of the b coefficients
        images: tensor of the images that are already reconstructed

    Returns:
        coefficients: tensor of the coefficients of the linear combination of the images
        image: tensor of the k-th image
    """

    k = directions.shape[0] 
    d = observation.shape[0]

    images = images.to(device)
    observation = observation.to(device)
    directions = directions.to(device)
    
    X = torch.cat((torch.transpose(images, 0,1), torch.zeros(k, k-1).to(device)), dim=0)
    b = torch.transpose(torch.cat((torch.zeros(d), b), dim=0).unsqueeze(0), 0, 1).to(device)
    Ia = torch.cat((torch.eye(d).to(device), directions), dim=0)
    A = torch.cat((X, b, Ia), dim=1).double()
    print("qua pure")
    B = torch.cat((observation, torch.zeros(k).to(device)), dim=0).double()

    print(f"Rank of A: {torch.linalg.matrix_rank(A.to('cpu'))}")
    print("e qua")
    # x = torch.linalg.lstsq(A, B, driver='gelsd')

    # coefficients = x[0][d:]
    # image = x[0][:d]

    x = torch.linalg.solve(A, B)
    coefficients = x[:-d]
    image = x[-d:]


    return coefficients, image


def get_observations(images, labels, model, criterion, optimizer, b, debug=False):

    """
    Get the observations in all the directions from model's update.
    
    Args:
        images(torch.Tensor): Flattened images tensor given in input to the model.
        labels(torch.Tensor): Labels corresponding to the images.
        model(nn.Module): Neural network model.
        criterion(nn.Module): Loss function.
        optimizer(torch.optim): Optimizer.
        b(torch.Tensor): Bias tensor to set.
        debug(bool) (optional): Debug flag. If True, the function returns a dictionary indicating the activations of each neuron. Default is False.

    Returns:
        obs(torch.Tensor): Observations in all the directions.
        dL_dA(torch.Tensor): Gradient of the loss with respect to the weights.
        dL_db(torch.Tensor): Gradient of the loss with respect to the bias.  
        acts(dict) (optional): Dictionary containing the activations of the neurons in the first layer of the model.
        
    """
    model.fc1.bias.data = b.detach().clone()
    model.train()
    optimizer.zero_grad()

    # register hook to keep track of the forward propagation (just for debugging purposes)
    if debug:
        acts = {}
        h = model.fc1.register_forward_hook(get_activations(acts))

    # forward pass
    preds = model(images)
    loss = criterion(preds, labels)

    # backward pass
    loss.backward()

    # get the gradient of the loss with respect to the activations
    dL_dA = model.fc1.weight.grad.data.detach().clone()
    dL_db = model.fc1.bias.grad.data.detach().clone()

    obs = dL_dA / dL_db.view(-1, 1)
    
    if debug:
        h.remove()

    if debug:
        return obs, dL_dA, dL_db, acts
    
    return obs, dL_dA, dL_db
    

def get_observation(images, labels, model, criterion, optimizer, b, direction, debug=False):

    """
    Get the observation in a specific directions from model's update.
    
    Args:
        images(torch.Tensor): Flattened images tensor given in input to the model.
        labels(torch.Tensor): Labels corresponding to the images.
        model(nn.Module): Neural network model.
        criterion(nn.Module): Loss function.
        optimizer(torch.optim): Optimizer.
        b(torch.Tensor): Bias tensor to set.
        direction(int): Direction to use.

    Returns:
        obs(torch.Tensor): Observations in all the directions.
        dL_dA(torch.Tensor): Gradient of the loss with respect to the weights.
        dL_db(torch.Tensor): Gradient of the loss with respect to the bias. 
    """
    model.fc1.bias.data = b.detach().clone()
    model.train()
    # optimizer.zero_grad()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer.zero_grad()

    # register hook to keep track of the forward propagation (just for debugging purposes)

    # forward pass
    preds = model(images)
    loss = criterion(preds, labels)

    # backward pass
    loss.backward()

    # get the gradient of the loss with respect to the activations
    dL_dA = model.fc1.weight.grad.data[direction].detach().clone()
    dL_db = model.fc1.bias.grad.data[direction].detach().clone()

    obs = (dL_dA / dL_db)

    if debug:
        # print(f"Loss value: {loss.item()}")
        print(f"Model bias: {model.fc1.bias.data}")
        print(f"dL/db control: {model.fc1.bias.grad.data[-1]}")
        print(f"Obs from get_observation: {obs}")
        return obs, dL_dA, dL_db
    
    return obs, dL_dA, dL_db


def check_predictions():
    images, labels = prepare_data()

    n_directions = 1
    image_size = images.shape[1]

    model = NN(input_dim=image_size, n_classes=10, hidden_dim=n_directions)
    model.fc2.bias.data = torch.tensor([10e2] * 10)
    model = model.double()

    acts = {}
    weights = []
    dL_db_list = []
    dL_dA_list = []
    loss_list = []
    rec_img_list = []
    pred_list = []

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()
    h = model.fc1.register_forward_hook(get_activations(acts))
    for i in range(images.shape[0]):
        optimizer.zero_grad()
        output = model(images[i])
        pred = torch.argmax(output)
        pred_list.append(pred)
        loss = criterion(output, labels[i])

        loss.backward()
        loss

    return pred_list

def check_multiple_observations(b_max, b, epsilon, images, labels, model, criterion, optimizer, direction, 
                                orth_subspace):
    """
    Check multiple observations in a specific direction.
    Args:
        b_min(torch.Tensor): Minimum bias tensor.
        b_max(torch.Tensor): Maximum bias tensor.
        epsilon(float): Step size.
        images(torch.Tensor): Flattened images tensor.
        labels(torch.Tensor): Labels tensor.
        model(nn.Module): Neural network model.
        criterion(nn.Module): Loss function.
        optimizer(torch.optim): Optimizer.
        b(torch.Tensor): Bias tensor.
        direction(int): Direction to use.
    """
    obs_list = []
    while b[direction] < b_max[direction]:
        print("----------------------")
        obs, dL_dA, dL_dB,   = get_observation(images, labels, model, criterion, optimizer, b, direction)
        obs_list.append(obs)
        orth_comp = project_onto_orth_subspace(obs, orth_subspace)
        print(f"Observation: {obs} | b_obs: {b[direction]} Norm: {torch.dot(orth_comp, orth_comp)}")

        weights_scaled, _, _ ,  obs_rec = check_real_weights(images, labels, model, direction, scale_factor=1)
        act_idx = [k[0] for k in weights_scaled]

        w = torch.tensor([k[1] for k in weights_scaled]).double()
        recon_obs  = w @ images[act_idx]
        print(f"Reconstructed obs from weights: {recon_obs}")

        b[direction] += epsilon

    return torch._stack(obs_list, dim=0)


def get_observation_batched(images, labels, model, direction, b):
    "Get the observation by computing per-sample gradients in a batched fashion"
    dL_db_list = []
    dL_dA_list = []

# TODO: Implement the function. Look at https://medium.com/pytorch/differential-privacy-series-part-2-efficient-per-sample-gradient-computation-in-opacus-5bf4031d9e22
# https://github.com/pytorch/opacus/blob/204328947145d1759fcb26171368fcff6d652ef6/opacus/grad_sample/linear.py
def get_observatation_no_batch(images, labels, model, direction, b):
    """
    Get the observation by computing per-sample gradients.
    """
    dL_db_list = []
    dL_dA_list = []

   
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()
    model.fc1.bias.data = b.detach().clone()

    for i in range(images.shape[0]):
        optimizer.zero_grad()
        pred = model(images[i])

        loss = criterion(pred, labels[i])
        loss.backward()
        dL_db = model.fc1.bias.grad[direction].detach().clone()
        dL_dA = model.fc1.weight.grad[direction].detach().clone()

        dL_db_list.append(dL_db)
        dL_dA_list.append(dL_dA)

    sum_dL_dB = sum(dL_db_list)
    dL_dA_tensor = torch.stack(dL_dA_list)
    obs_rec = torch.sum(dL_dA_tensor, dim=0)/sum_dL_dB
    return  obs_rec, torch.sum(dL_dA_tensor, dim=0), sum_dL_dB

def get_observations_no_batch(images, labels, model, current_b):
    """
    Get the observation by computing sequentially oer per-sample gradients.
    """

    optimizer = torch.optim.SGD(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model.train()
    model.fc1.bias.data = current_b.detach().clone()

    for i in range(images.shape[0]):
        optimizer.zero_grad()
        pred = model(images[i])

        loss = criterion(pred, labels[i])
        loss.backward()
        dL_db_image = model.fc1.bias.grad.data.detach().clone()
        dL_dA_image = model.fc1.weight.grad.data.detach().clone()

        if i == 0:
            dL_dA_all = dL_dA_image.unsqueeze(0)
            dL_db_all = dL_db_image.unsqueeze(0)

        else:
            dL_dA_all = torch.cat((dL_dA_all, dL_dA_image.unsqueeze(0)), 0)
            dL_db_all = torch.cat((dL_db_all, dL_db_image.unsqueeze(0)), 0)
        
    sum_dL_dB = torch.sum(dL_db_all, dim=0).view(-1, 1)
    obs_rec = torch.sum(dL_dA_all, dim=0)/sum_dL_dB

    return obs_rec, sum_dL_dB, 

def check_search_direction(observation_history, max_norm, min_norm):
    
    """
    Check if an observation is in the same span of the previous ones by comparing the norm of the orthogonal component."""
    
    last_obs = observation_history[-1]
    norm_history = [i[2] for i in observation_history]

    # Case 1: we are moving from right, compare the current norm with the one at the previous step. If the ratio is larger than compared to b_min, probably fake observation
    if last_obs[0] > observation_history[-2][0]:
        if norm_history[-2] / last_obs[2] <= last_obs[2] / min_norm:
            print("Warning: there might not be an image here")
            return True
        else:
            return False
    # Case 2: we are moving from left, compare the current norm with the norm at b_min and the norm at b_max. If the ratio is larger than at the current b_max, probably real obs
    else:
        if last_obs[2] / min_norm < max_norm / last_obs[2]:
            print("Warning: There might not be an image here")
            return True
        else:
            return False
    

def find_strips(images, labels, n_classes, n_directions, control_bias=1e5, classification_weights_value=1e10, classification_bias_value=0.1, threshold=1e-5,
                 noise_norm=None, epsilon=None):
    # TODO: improve documentation
    """
    Find observation points in all the directions.
    Args:
        images(torch.Tensor): Flattened images tensor.
        labels(torch.Tensor): Labels tensor.
        n_classes(int): Number of classes in the dataset.
        n_directions(int): Number of directions to consider.
        
        Returns:
            strips(dict): Dictionary containing the location of the observations.
            
        """
    image_size = images.shape[1]

    strips = {i: [] for i in range(images.shape[0])}
    orth_subspaces = {}
    alphas = {i: [] for i in range(images.shape[0])}

    observations_history = {i: dict() for i in range(n_directions)}
    for i in observations_history.keys():
        observations_history[i] = {j: [] for j in range(images.shape[0])}
    
    model = NN(input_dim=image_size, n_classes=n_classes, hidden_dim=n_directions + 1)
    model = model.double()

    # bias that controls outputs distribution
    model.fc1.bias.data[n_directions:] = control_bias
    # classification layer modeifications to keep the output controlled by the previous bias
    if classification_weights_value is not None:
        model.fc2.weight.data = (torch.ones_like(model.fc2.weight) * classification_weights_value).double()

    if noise_norm is not None:
        noise = torch.randn_like(model.fc2.weight) * noise_norm
        model.fc2.weight.data += noise
    if classification_bias_value is not None:
        model.fc2.bias.data = (torch.ones_like(model.fc2.bias) * classification_bias_value).double()

    b_tensor = - torch.matmul(model.fc1.weight, torch.transpose(images, 0, 1))

    # useful for checking the binary search accuracy
    b_sorted, indices = torch.sort(torch.tensor(b_tensor.detach().clone()), dim=1)
    
    if epsilon is None:
        epsilon = torch.min(torch.abs(b_sorted[:, 1:] - b_sorted[:, :-1]), 1).values
    else:
        epsilon = torch.ones_like(b_sorted[0]).double() * epsilon

    # NOTE: Check this. Now using a version that is artificial
    b_min = torch.min(b_sorted, dim=1).values  - 100
    b_max = torch.max(b_sorted, dim=1).values  + 100
    B_max = b_max.detach().clone()
    interval = b_max - b_min

    for j in range(images.shape[0]):
        print(f"------Round {j}------")     
        init_time = time.time()
        current_b = b_min + (b_max - b_min) / 2
        current_b[n_directions:] = control_bias

        for i in range(n_directions):
            image_start_time = time.time()
            print(f"-----Direction {i}-----")
            found_image = False
            # if we have found the first image, we can initialize the orthogonal subspace

            while not found_image:
                # 1. Get the observation 
                current_b[i] = b_min[i].item() + (b_max[i].item() - b_min[i].item()) / 2
                obs, dL_dA, dL_db = get_observatation_no_batch(images, labels, model, i, current_b)
                
                # 2. Check the stopping condition
                if (interval[i] < epsilon[i] / 2):
                    if len(strips[i]) == 0:
                        if dL_db == 0:
                            current_b[i] = b_max[i].item()
                            obs, dL_dA, dL_db = get_observatation_no_batch(images, labels, model, i, current_b)
                            obs_max = obs.detach().clone()
                            observations_history[i][j].append((current_b[i], obs, 0))
                        orth_subspaces[i] = obs.unsqueeze(0)

                    else:
                        time_span = time.time()
                        flag, norm = check_span(orth_subspaces[i], obs, norm_threshold=threshold, debug=False)
                        print(f"Time to check span: {time.time() - time_span}")
                        if flag:
                            current_b[i] = b_max[i].item()
                            obs, dL_dA, dL_db = get_observatation_no_batch(images, labels, model, i, current_b)
                            obs_max = obs.detach().clone()
                            print(f"current b:  {current_b[i].item()}  | b_min: {b_min[i].item()} | b_max: {b_max[i].item()} | Real position {b_sorted[i, j].item()}")
                            flag_check, norm__check = check_span(orth_subspaces[i], obs, norm_threshold=threshold, debug=True)
                            if flag_check:
                                raise ValueError("The observation is in the span, something is wrong")
                        orth_comp = project_onto_orth_subspace(obs, orth_subspaces[i])
                        orth_subspaces[i] = torch.cat((orth_subspaces[i], orth_comp.unsqueeze(0)), 0)

                    strips[i].append((current_b[i], obs))
                    found_image = True
                    weights, _ = check_real_weights(images, labels, model, i, scale_factor=1, debug=False)
                    alphas[i].append(weights)

                    print(f"Found image at position {current_b[i].item()} ")
                    print(f"Time to find the image: {time.time() - image_start_time}")
                    if abs(current_b[i].item() - b_sorted[i, j].item()) > epsilon[i]:
                        raise ValueError("Warning: Error in the binary search")
                    break
                

                # 3. Check in which direction to move
                # search for the first image
                if len(strips[i]) == 0:
                    if (dL_dA != 0.).any().int() == 0:
                        # no image is activated, we move right
                        b_min[i] = current_b[i].item()
                        obs_min = obs.detach().clone()
                    else:
                        # active neurons, we  move left
                        b_max[i] = current_b[i]
                        obs_max = obs.detach().clone()
                    observations_history[i][j].append((current_b[i], obs, 0))
                else:
                    print(f"current b:  {current_b[i].item()}  | b_min: {b_min[i].item()} | b_max: {b_max[i].item()} | Real position {b_sorted[i, j].item()}")
                    flag, norm = check_span(orth_subspaces[i], obs, norm_threshold=threshold, debug=True)

                    #NOTE: case to debug
                    if current_b[i] > b_sorted[i, j].item() and flag:
                        print("Warning: this observation should not be in the span, however it is")
                        _, obs_debug = check_real_weights(images, labels, model, i, debug=False)
                        assert torch.equal(obs, obs_debug), "Observations are not equal, something is wrong"

                    if flag:
                        # the observation is a linear combination of the orthogonal basis. We move right
                        b_min[i] = current_b[i].item()
                        obs_min = obs.detach().clone()
                    else:
                        # the observation is not a linear combination of the orthogonal basis. We move left
                        b_max[i] = current_b[i].item()
                        obs_max = obs.detach().clone()
                    observations_history[i][j].append((current_b[i], obs, norm))
                
                interval[i] = b_max[i].item() - b_min[i].item()
            
            # 5. Once the image id found, we update the search interval for the next round
            b_min[i] = b_max[i].item()
            b_max[i] = B_max[i].item()
            interval[i] = b_max[i].item() - b_min[i].item()
        print(f"Time to find the images: {time.time() - init_time}")

    return strips, alphas, model.fc1.weight.data, model.fc1.bias.data


def find_strips_parallel(images, labels, n_classes, n_directions, control_bias=1e5, classification_weights_value=1e-1, classification_bias_value=0.1, threshold=1e-5,
                 noise_norm=None, epsilon=1e-8, device='cpu'):
    
    """
    Find observation points in all the directions. This version uses a parallel search for each direction.

    Args:
        images(torch.Tensor): Flattened images tensor.
        labels(torch.Tensor): Labels tensor.
        n_classes(int): Number of classes in the dataset.
        n_directions(int): Number of directions to consider.
        control_bias(float): Bias value to control the output distribution.
        classification_weights_value(float): Value of the classification weights.
        classification_bias_value(float): Value of the classification bias.
        threshold(float): Threshold value to check the norm of the orthogonal component.
        noise_norm(float): Noise value to add to the classification weights.
        epsilon(float): stopping threshold for the binary search.
        
        Returns
            strips_obs(dict): Tensor containing the observations.
            strips_b(dict): Tensor containing the position corresponding to the observations.
            neurons(torch.Tensor): Weights of the neural network.
            bias(torch.Tensor): Bias of the neural network.
    """

    image_size = images.shape[1]
    images = images.to(device)
    labels = labels.to(device)
    
    strips_obs = torch.zeros( n_directions, n_directions + 1, images.shape[1]).double().to(device)
    strips_b = torch.zeros(n_directions, n_directions + 1).double().to(device)
    orth_subspaces = torch.zeros(n_directions, n_directions + 1, images.shape[1]).double().to(device)
    alphas = {i: [] for i in range(images.shape[0])}

    observations_history = {i: dict() for i in range(n_directions)}
    for i in observations_history.keys():
        observations_history[i] = {j: [] for j in range(images.shape[0])}
    
    model = NN(input_dim=image_size, n_classes=n_classes, hidden_dim=n_directions + 1)
    model = model.double().to(device)

    # bias that controls outputs distribution
    model.fc1.bias.data[n_directions:] = control_bias
    # classification layer modeifications to keep the output controlled by the previous bias
    if classification_weights_value is not None:
        model.fc2.weight.data = (torch.ones_like(model.fc2.weight) * classification_weights_value).double()

    if noise_norm is not None:
        noise = torch.randn_like(model.fc2.weight) * noise_norm
        model.fc2.weight.data += noise
    if classification_bias_value is not None:
        model.fc2.bias.data = (torch.ones_like(model.fc2.bias) * classification_bias_value).double()

    b_tensor = - torch.matmul(model.fc1.weight, torch.transpose(images, 0, 1))

    # useful for checking the binary search accuracy
    b_sorted, indices = torch.sort(torch.tensor(b_tensor.detach().clone()), dim=1)

    epsilon = torch.ones_like(model.fc1.bias.data).double() * epsilon

    # TODO: Check this. Now using a version that is artificial
    b_min = torch.min(b_sorted, dim=1).values  - 100
    b_max = torch.max(b_sorted, dim=1).values  + 100
    B_max = b_max.detach().clone()
    interval = b_max - b_min

    for j in range(images.shape[0]):
        image_time = time.time()
        print(f"------Round {j}------")
        # mask to keep track of the directions where we have already found the image
        search_dir = torch.ones(n_directions+1).bool().to(device)
        search_dir[-1] = False

        while not torch.all(~search_dir):

            current_b = b_min.detach().clone() + (b_max.detach().clone() - b_min.detach().clone()) / 2
            current_b[n_directions:] = control_bias

            observations, dL_db = get_observations_no_batch(images=images,
                                                          labels=labels,
                                                          model=model,
                                                          current_b=current_b)
            
            stop_search_mask = interval < epsilon / 2
            stop_search_mask[-1] = False
            if torch.any(stop_search_mask):
                stop_indices = torch.nonzero(stop_search_mask, as_tuple=False)
                
                if j == 0:
                    # first image, we need to ensure to stop on the left of the image
                    non_active_mask = dL_db[stop_search_mask] == 0
                    if torch.any(non_active_mask):
                        # no image is activated, we move right
                        current_b[stop_indices[non_active_mask]] = b_max[stop_indices[non_active_mask]]
                        observations, _ = get_observations_no_batch(images=images,
                                                                    labels=labels,
                                                                    model=model,
                                                                    current_b=current_b)
                        for i in range(stop_search_mask.shape[0]):
                            if stop_search_mask[i] and non_active_mask[i]:
                                observations_history[i][j].append((current_b[i].detach().clone(), observations[i].detach().clone(), 0))

                else:
                    # TODO: code this case (stopped right to the real image)
                    for i in range(stop_search_mask.shape[0]):
                        if stop_search_mask[i]:
                            flags[i], _ = check_span(orth_subspaces[i, :j], observations[i], norm_threshold=threshold, debug=False)
                    if torch.any(flags == 1):
                        current_b = torch.where(flags == 1, b_max, current_b)
                        observations, _ = get_observations_no_batch(images=images,
                                                                    labels=labels,
                                                                    model=model,
                                                                    current_b=current_b)
                        for i in range(stop_search_mask.shape[0]):
                            if stop_search_mask[i].item() is True and flags[i] == 1:
                                observations_history[i][j].append((current_b[i].detach().clone(), observations[i].detach().clone(), 0))
                                flags[i], _ = check_span(orth_subspaces[i, :j], observations[i], norm_threshold=threshold, debug=False)
                                if flags[i]:
                                    raise ValueError("Warning: the observation is in the span, something is wrong")
                                
                        for i in range(stop_indices.shape[0]):
                            orth_comp = project_onto_orth_subspace(observations[i], orth_subspaces[i, :j])
                            orth_subspaces[i, j, :] = orth_comp.detach().clone()

                
                strips_b[j, stop_indices] = current_b[stop_indices].detach().clone()
                strips_obs[j, stop_indices, :] = observations[stop_indices].detach().clone()
                search_dir[stop_indices] = False
                search_dir[-1] = False

                for i in stop_indices:
                    print(f"Found image in direction {i.item()} at position {current_b[i].item()} | Real position {b_sorted[i, j].item()}")
                    if abs(current_b[i].item() - b_sorted[i, j].item()) > epsilon[i]:
                        raise ValueError("Warning: Error in the binary search")

                if torch.all(~search_dir):
                    break


            # check in which direction to move
            # search for the first image in each direction (mask found_in_all_dir helps to skip the directions where we have already found the image)
            if j == 0:
                b_min = torch.where(dL_db.squeeze() * search_dir.double() == 0, current_b, b_min)
                b_max = torch.where(dL_db.squeeze() * search_dir.double() != 0, current_b, b_max)
                
                for i in range(search_dir.shape[0]):
                    if search_dir[i]:
                        observations_history[i][j].append((current_b[i].detach().clone(), observations[i].detach().clone(), 0))
            else:
                # TODO: now it is sequential, it might be parallelized, but it is not clear how to do it and if it is worth it
                flags = (torch.ones(n_directions + 1) * -1).to(device)
                for i in range(search_dir.shape[0] -1):
                    if search_dir[i]:
                        flags[i], _ = check_span(orth_subspaces[i, :j], observations[i], norm_threshold=threshold, debug=False)
                        if current_b[i] > b_sorted[i, j].item() and flags[i]:
                            raise ValueError("Warning: this observation should not be in the span, however it is")

                b_min = torch.where(flags == 1, current_b,  b_min)

                b_max = torch.where(flags == 0, current_b,  b_max)

                for i in range(search_dir.shape[0]):
                    if search_dir[i]:
                        observations_history[i][j].append((current_b[i], observations[i], 0)) 

            interval = b_max - b_min
        if j == 0:
            orth_subspaces[:, 0, :] = observations[:-1, :].detach().clone() # we do not need the last observation (from the control bias)
        else:
            for i in range(n_directions):
                orth_comp = project_onto_orth_subspace(observations[i], orth_subspaces[i, :j])
                orth_subspaces[i, j, :] = orth_comp.detach().clone()
    
        b_min = b_max.detach().clone()
        b_max = B_max.detach().clone()
        interval = b_max - b_min

        print(f"Time to find the images: {time.time() - image_time}")

    return strips_obs, strips_b, model.fc1.weight.data, model.fc1.bias.data


def find_corresponding_strips(strips_obs, strips_b, threshold=1e-5):
    """
    Find the corresponding strips in the orthogonal subspace.
    Args:
        strips_obs(torch.Tensor): Tensor containing the observations.
        strips_b(torch.Tensor): Tensor containing the position corresponding to the observations.
        
        Returns:
            corresponding_strips_obs(torch.Tensor): Tensor containing the corresponding observations.
            corresponding_strips_b(torch.Tensor): Tensor containing the corresponding positions.
    """

    strips_dict = {i: [(strips_b[j, i], strips_obs[j, i]) for j in (range(strips_obs.shape[0])) ] for i in range(strips_obs.shape[1])}
    v = strips_dict[0] # fix direction v
    v_orth = v[0][1].unsqueeze(0) # first element of the orthogonal supspace spanned by v
    strips_dict.pop(0)

    for i in range(len(v)):
        if i != 0:
            residual = project_onto_orth_subspace(v[i][1], v_orth)
            v_orth = torch.cat((v_orth, residual.unsqueeze(0)), 0)

    corresponding_obs = {i: v[i][1].unsqueeze(0) for i in range(len(v))}
    corresponding_bias = {i: [v[i][0].item()] for i in range(len(v))}
    for d in strips_dict.keys():
        u = strips_dict[d]
        for i in range(v_orth.shape[0]):
            curr_orth = v_orth[:i + 1] # select the first i orthogonal components in v_orth (corresponding to the component at the i-th observation)

            not_found = True
            j = 0

            while not_found:
                # project the observation u[j] onto the orthogonal subspace spanned by the first i orthogonal components of v
                residual = project_onto_orth_subspace(u[j][1], curr_orth)
                # if the residual is zero, then the observation u[j] is in the span of the orthogonal subspace
                if torch.dot(residual, residual) < threshold:
                    # add the observation to the corresponding observation list
                    corresponding_obs[i] = torch.cat((corresponding_obs[i], u[j][1].unsqueeze(0)), 0)
                    corresponding_bias[i].append(u[j][0].item())
                    not_found = False
                    corr = u.pop(j)
                    assert torch.equal(corr[1],corresponding_obs[i][-1])


                else:
                    # add the residual to the orthogonal subspace
                    curr_orth = torch.cat((curr_orth, residual.unsqueeze(0)), 0)
                    j += 1
                    if j == len(u):
                        not_found = False
                        raise ValueError(f"Warning: an image correspondence has not been detected even though it should")
                        # no_corr_list.append(v[i])
                        # for i in range(curr_orth.shape[0]):
                        #     print(f"Dot product with the orthogonal component {i}: {torch.dot(residual, curr_orth[i])}")

    return corresponding_obs, corresponding_bias
        