import torch
import torchvision
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import itertools

import matplotlib.pyplot as plt

import bisect
import random
import math
import argparse
import os

from utils_search import *
from utils_misc import *
from utils_test import * 

def main():

    args = parse_args()

    set_seeds(3)

    images, all_labels = prepare_data()

    num_samples = args.n_samples
    sample = np.random.choice(range(images.shape[0]), size=num_samples, replace=False)
    images_client = images[sample]
    labels_client = all_labels[sample]

    n_directions = args.n_samples

    # STEP 1 + 2
    
    # strips_dict, alphas, neurons, bias = find_strips(images=images_client,
    #                                             labels=labels_client,
    #                                             n_classes=10,
    #                                             n_directions=n_directions,
    #                                             classification_weights_value=1e-5,
    #                                             classification_bias_value=1e-3,
    #                                             control_bias=500,
    #                                             noise_norm=0,
    #                                             epsilon=1e-1,
    #                                             threshold=1e-6)

    strips_obs, strips_b, neurons, bias = find_strips_parallel(images=images_client,
                                                labels=labels_client,
                                                n_classes=10,
                                                n_directions=n_directions,
                                                classification_weights_value=1,
                                                classification_bias_value=1e-5,
                                                control_bias=5e5,
                                                noise_norm=0,
                                                epsilon=1e-5,
                                                threshold=1e-6,
                                                directions_weights_value=1,
                                                device=args.device)
                                                
    
    # STEP 3: find corresponding strips
    time_corr = time.time()
    # corresponding_strips, corresponding_bias = find_corresponding_strips_sequential(strips_dict)   
    corresponding_strips, corresponding_bias = find_corresponding_strips(strips_obs, strips_b)   
    print(f"Time to find corresponding strips: {time.time() - time_corr}")

    # STEP 4: reconstruct images
    recon_images = corresponding_strips[0][0].unsqueeze(0)

    # for k in corresponding_strips.keys():
    #     restore_image(corresponding_strips[k][0].unsqueeze(0), display=True)
    # restore_image(recon_images[0][0], display=True)
    

    
    # show_true_images(images_client)
    for k in range(1, num_samples):
        # pick the k-the observation in direction 0
        obs = corresponding_strips[k][0]
        # pick the first k directions
        a = neurons[:k+1, :]
        # pick the associated kth-b coefficients
        b = torch.tensor(corresponding_bias[k][:k+1]).double()
        time_lin_sys = time.time()
        coefficients, image = solve_linear_system(obs, a, b, recon_images, device=args.device)
        print(f"Time to solve linear system: {time.time() - time_lin_sys}")
        print(f"coefficients: {coefficients} | sum: {torch.sum(coefficients)}")
        image = image / coefficients[-1]
        recon_images = torch.cat((recon_images, image.unsqueeze(0)), dim=0)
    if args.display:
        print("Showing reconstructed images")
        for k in range(num_samples):
            restore_images([recon_images[k], images_client[k]], device=args.device, display=True, title=f"Reconstructed image {k}")

    # recon_images = reconstruct_images(corresponding_strips, images_client, n_directions)
if __name__ == '__main__':
    main()