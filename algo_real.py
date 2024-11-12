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

import itertools

import matplotlib.pyplot as plt

import bisect
import random
import math
import argparse
import os

from utils import *



# def find_strips(images, labels, n_classes, n_directions):
#     """
#     Applies binary search to locate images in the space. 
#     Steps:
#     1. Initialize a set of random weights for the first layer of the neural network.
#     2. forward pass the images through the network. (In a real FL setting, this would be done by each client)
#     4. Apply binary search to locate sequentially all the images. ( In a FL setting, this means having multiple communication rounds)
#     5. Return the observations and the corresponding b values.

#     Args:
#         images(torch.Tensor): Images to be searched.  
#         labels(torch.Tensor): Labels of the images.
#         n_classes(int): Number of classes in the dataset.
#         n_directions(int): Number of directions to search.
#     Returns:
#         strips(dict): Dictionary containing the location of the observations.
#         strips_b(dict): Dictionary containing the b values corresponding to each observation.
#     """
#     image_size = images.shape[1]

#     # acts = {}
#     strips = {i: [] for i in range(images.shape[0])}
#     orth_subspaces = {}

#     observations_history = {i: dict() for i in range(n_directions)}
#     for i in observations_history.keys():
#         observations_history[i] = {j: [] for j in range(images.shape[0])}
    
#     model = NN(input_dim=image_size, n_classes=n_classes, hidden_dim=n_directions + 1)
#     model = model.double()

#     # bias to control the output of the model
#     model.fc1.bias.data[n_directions:] = 1e5

#     # classification layer to keep the output controlled by the previous bias
#     model.fc2.bias.data = torch.tensor([0.1] * 10).double()
#     model.fc2.weight.data = (torch.ones_like(model.fc2.weight) * 1e10).double()

    
#     # b_1 = - torch.matmul(model.fc1.weight, (2 * torch.ones(image_size, 1)))
#     # b_2 = torch.matmul(model.fc1.weight, (2 * torch.ones(image_size, 1)))

#     b_tensor = - torch.matmul(model.fc1.weight, torch.transpose(images, 0, 1))

#     # useful for checking how good is the binary search
#     b_sorted, indices = torch.sort(torch.tensor(b_tensor.detach().clone()), dim=1)

#     epsilon = torch.min(torch.abs(b_sorted[:, 1:] - b_sorted[:, :-1]), 1).values
#     # epsilon = torch.tensor([0.01] * b_tensor.shape[0])


#     # NOTE: Check this. Now using a version that is artificial
#     b_min = torch.min(b_sorted, dim=1).values  - 300
#     b_max = torch.max(b_sorted, dim=1).values  + 300
#     B_max = b_max.detach().clone()
#     interval = b_max - b_min

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#     for j in range(images.shape[0]):
#         print(f"------Round {j}------")        

#         thresholds = torch.tensor([1e-5] * n_directions).double()

#         # mask  to keep track of the images that have been found
#         found_all = torch.zeros(n_directions) 

#         while not (torch.equal(found_all, torch.ones(n_directions))):
            
#             current_b = b_min + (b_max-b_min)/2 # find the middle point

#             obs, dL_dA, dL_db = get_observations(images=images,
#                                                 labels=labels,
#                                                 model=model,
#                                                 criterion=criterion,
#                                                 optimizer=optimizer,
#                                                 b=current_b
#                                                 )

#             # for i in range(n_directions):
#             #     observations_history[i][j].append((current_b[i], obs[i]))

#             # # check the number of activations
#             # for_act_mask = []
#             # for i in acts.keys():
#             #     for_act_mask.append(acts[i] > 0)
#             # act_tens = torch.stack(for_act_mask, dim=0)
#             # n_act = torch.sum((act_tens > 0).int(), dim=1)
            
#             #NOTE : Debugging purposes
#             # print(f"Number of activations: {n_act}")

#             act_mask = (dL_dA != 0.).any(dim=1).int()

#             for i in range(n_directions):

#                 # NOTE: Debugging purposes

#                 if (interval[i] < epsilon[i] / 2) and (found_all[i] != 1):
                    
#                     if dL_db[i] == 0:
#                         current_b[i] = b_max[i]
#                         obs[i] , dL_dA[i], dL_db[i] = get_observatation_no_batch(images=images, 
#                                                                labels=labels,
#                                                                model=model,
#                                                                b=current_b,
#                                                                direction=i
#                                                                )
#                         _, obs[i] = check_real_weights(images, labels, model, i, debug=False)
                
#                         observations_history[i][j].append(( current_b[i], obs[i], 0))
#                         obs_max = obs[i].detach().clone()
                    
#                     if len(strips[i]) >= 1:
#                         # NOTE: Debugging purposes
#                         # orth_comp = project_onto_orth_subspace(obs[i], orth_subspaces[i])
#                         # for k in range(len(orth_subspaces[i])):
#                             # print(f"Dot product with previous comp. {k}: {torch.dot(orth_subspaces[i][k], orth_comp)}")
#                         flag, norm = check_span(orth_subspaces[i], obs[i], norm_threshold=max(thresholds[i], 1e-6), debug=True)
#                         print(f"current b: {current_b[i]} | epsilon: {epsilon[i]} | Real position {b_sorted[i, j].item()}")
#                         print("---")
#                         observations_history[i][j].append((current_b[i], obs[i], norm))
                        
#                         from_left = False
#                         # add additional steps on the right, in case the algorithm stops before finding the image
#                         while (flag) and current_b[i] < B_max[i]:
#                             b_min[i] = current_b[i]
#                             current_b[i] = b_min[i] + epsilon[i]/2

#                             obs[i] , dL_dA[i], dL_db[i] = get_observatation_no_batch(images=images, 
#                                                                labels=labels,
#                                                                model=model,
#                                                                b=current_b,
#                                                                direction=i
#                                                                )
#                             obs_max = obs[i].detach().clone()

#                             print("Checking again the span after additional step...")
#                             # orth_comp = project_onto_orth_subspace(obs[i], orth_subspaces[i])
#                             # for k in range(len(orth_subspaces[i])):
#                                 # print(f"Dot product with previous comp. {k}: {torch.dot(orth_subspaces[i][k], orth_comp)}")
#                             flag, norm = check_span(orth_subspaces[i], obs[i], norm_threshold=thresholds[i], debug=True)
#                             if j < (b_sorted.shape[1] - 1):
#                                 print(f"current b: {current_b[i]} | Real position {b_sorted[i, j].item()} | Previous Image: {b_sorted[i, j-1].item()}| Next Image: {b_sorted[i, j+1].item()}| epsilon: {epsilon[i]}  ")
#                             else:
#                                 print(f"current b: {current_b[i]} | Real position {b_sorted[i, j].item()} | Previous Image: {b_sorted[i, j-1].item()}|  epsilon: {epsilon[i]}  ")

#                             observations_history[i][j].append((current_b[i], obs[i], norm))
#                             from_left = True

#                         if from_left:
#                             """Additional check. If we stop at a fake observation, we need to keep moving right until we found a real norm difference"""
#                             if (observations_history[i][j][-1][2]/ observations_history[i][j][-2][2] < 10) or \
#                                 (observations_history[i][j][-1][2] < 10 * thresholds[i]):
#                                 print("Probably a fake observation, looking in obs history...")
#                                 for k in range(len(observations_history[i][j]) - 1, 0, -1):
#                                     if observations_history[i][j][k - 1][2] / observations_history[i][j][k][2] > 10:
#                                         print(f"Norm ratio: {observations_history[i][j][k - 1][2] / observations_history[i][j][k][2]}")
#                                         print(f"Norm at possible observation: {observations_history[i][j][k - 1][2]} | {observations_history[i][j][k][2]}")

#                                         # TODO: invece di prendere direttamente il massimo, prova a confrontare, scorrendo le osservazioni, il rapporto tra norma corrente e precedente
#                                         # e norma corrente e successiva. Se la differenza tra i ratei e` ampia, allora vuol dire che siamo passati da un punto in cui non eravamo nello span
#                                         # a uno in cui siamo nello span, per cui prendiamo il punto corrente.
#                                         thresholds[i] = 2 * max(observations_history[i][j][k][2], thresholds[i])
#                                         b_min[i] = observations_history[i][j][k][0]
#                                         b_max[i] = observations_history[i][j][k - 1][0]
#                                         current_b[i] = b_max[i]
#                                         print(f"New interval: b_min {b_min[i]} | b_max {b_max[i]} | Real position {b_sorted[i, j].item()}")
#                                         print("New threshold: ", thresholds[i])
#                                         obs[i] , dL_dA[i], dL_db[i] = get_observatation_no_batch(images=images, 
#                                                                labels=labels,
#                                                                model=model,
#                                                                b=current_b,
#                                                                direction=i
#                                                                )
#                                         obs_max = obs[i].detach().clone()
#                                         break
                            
#                         # we should check if a "fake observation" is found
#                         if not(flag) and not(from_left):
#                             """ 
#                             if there is a new observed image, we expect the ratio between the norms at b_min and the one at
#                             current_b to be large. If they are very close, it means that there is no point between b_min
#                             and current_b, so there is a fake observation. In this case, we move right and continue the search starting 
#                             from the first observation where we see a significative norm difference between consecutive observations.
#                             """
#                             norm_history = [x[2] for x in observations_history[i][j]]
#                             b_history = [x[0] for x in observations_history[i][j]]

#                             _, b_min_norm = check_span(orth_subspaces[i], strips[i][-1][1], norm_threshold=thresholds[i], debug=True)

#                             if (norm > thresholds[i]) and (norm / b_min_norm < 10):
#                                 # NOTE: print for debugging purposes
#                                 print(f"Fake observation found at {current_b[i]} | Real position {b_sorted[i, j].item()}")
#                                 print(f"Norm at b_min: {b_min_norm} | Norm at current_b: {norm}")

#                                 # print(f"New interval: b_min {b_min[i]} | b_max {b_max[i]} | Real position {b_sorted[i, j].item()}")
#                                 for k in range(len(norm_history) - 1, 0, -1):
#                                     if norm_history[k - 1] / norm_history[k] > 2 and norm_history[k - 1]  > 10 * thresholds[i]:
#                                         print(f"Norm ratio: {norm_history[k - 1] / norm_history[k]}")
#                                         print(f"Norm at possible observation: {norm_history[k - 1]} | {norm_history[k]}")
#                                         norm_ratio = norm_history[k - 1] / norm_history[k]
#                                         break
#                                         # thresholds[i] = norm_history[k] + (0.25 * (norm_history[k - 1] - norm_history[k]))
#                                 print("The largest ratio between consecutive norms is: ", max([norm_history[k - 1] / norm_history[k] for k in range(len(norm_history) - 1)]))
                                
#                                 """
#                                 Once we find a possible interval, we keep searching by looking the norm ratio between the current obs and the 
#                                 b_min and b_max. If the norm is similar to 1, it means there is nothing in between, otherwise, it means that there might
#                                 be an image. 
#                                 """
#                                 b_min[i] = b_history[k]
#                                 b_max[i] = b_history[k - 1]

#                                 norm_min = norm_history[k]

#                                 current_b[i] = b_min[i] + (b_max[i] - b_min[i]) / 2
#                                 interval[i] = b_max[i] - b_min[i]
#                                 current_b[i] = b_max[i]

#                                 thresh_norm_ratio = norm_history[k] / b_min_norm
#                                 keep_searching = True

#                                 while keep_searching:
#                                     print(f"New interval: b_min {b_min[i]} | b_max {b_max[i]} | Real position {b_sorted[i, j].item()}")
#                                     # print("New threshold: ", thresholds[i])


#                                     obs[i] , dL_dA[i], dL_db[i] = get_observatation_no_batch(images=images, 
#                                                             labels=labels,
#                                                             model=model,
#                                                             b=current_b,
#                                                             direction=i
#                                                             )
#                                     _, cur_norm = check_span(orth_subspaces[i], obs[i], norm_threshold=thresholds[i], debug=True)
#                                     observations_history[i][j].append((current_b[i], obs[i], norm))

#                                     # Case 1: no new images on the left side of the current observation, we move right
#                                     if cur_norm / norm_min < 10:
#                                         print(f"Case 1: Moving right | Norm ratio: {cur_norm / norm_min}")

#                                         b_min[i] = current_b[i]
#                                         interval[i] = b_max[i] - b_min[i]
#                                         current_b[i] = b_min[i] + interval[i] / 2
#                                         print(f"New interval: current b {current_b[i]} | b_min {b_min[i]} | b_max {b_max[i]} | Real position {b_sorted[i, j].item()}") 

                                    
#                                     # Case 2: One or more images on the left, we move left
#                                     elif cur_norm / norm_min >= 10:
#                                         print(f"Case 2: Moving left | Norm min: {norm_min} | Norm current: {cur_norm}")

#                                         b_max[i] = current_b[i]
#                                         interval[i] = b_max[i] - b_min[i]
#                                         current_b[i] = b_min[i] + interval[i] / 2
#                                         print(f"New interval: current b {current_b[i]} | b_min {b_min[i]} | b_max {b_max[i]} | Real position {b_sorted[i, j].item()}") 
                                    
#                                     else:
#                                         print("Warning, no images on the left, and no images on the right")

#                                     if interval[i] < epsilon[i] / 2:
#                                         keep_searching = False
#                                         obs_max, _, _ = get_observatation_no_batch(images=images, 
#                                                             labels=labels,
#                                                             model=model,
#                                                             b=b_max,
#                                                             direction=i
#                                                             )
#                                         obs[i] = obs_max
#                                         observations_history[i][j].append((b_max[i], obs_max, 0))


#                     if len(strips[i]) > 0:
#                         orth_comp = project_onto_orth_subspace(obs[i], orth_subspaces[i])
#                         orth_subspaces[i] = torch.cat((orth_subspaces[i], orth_comp.unsqueeze(0)), 0)
#                     # NOTE: debug
#                     # if len(strips[i]) > 0:
#                     #     _, new_thresh = check_span(orth_subspaces[i], obs[i], norm_threshold=thresholds[i], debug=True)
#                     #     thresholds[i] = new_thresh
#                     print()

#                     # additional check to see if everything is working well:
#                     if j >= 1:
#                         norm_history = [x[2] for x in observations_history[i][j]]
#                         prev_norm_list = []
#                         for k in range(len(norm_history) - 1, 0, -1):
#                             if norm_history[k - 1] / norm_history[k] <= 10:
#                                 prev_norm_list.append(norm_history[k])
#                             else:
#                                 break
#                         if len(prev_norm_list) > 0:
#                             mean_norm = torch.mean(torch.tensor(prev_norm_list))
#                             if observations_history[i][j][-1][2] / mean_norm > 10:
#                                 print("Warning: probably fake observation found, you should restart the search")
                                
#                     strips[i].append((b_max[i].detach().clone(), obs_max.detach().clone())) 
#                     found_all[i] = 1

#                     # if j > 0:
#                         # for k in range(len(orth_subspaces[i])):
#                         #     print(f"Dot product with previous comp. {k}: {torch.dot(orth_subspaces[i][k], orth_comp)}")
#                     #  | Next image: {b_sorprint(f"Found image at {b_max[i]} | Real position {b_sorted[i, j].item()}ted[i, j + 1].item()}")
                    
#                     print(f"Found image at {b_max[i]} | Real position {b_sorted[i, j].item()}")    
#                     print(check_real_weights(images, labels, model, i, debug=False, display_weights=True))                
#                     print()


#                 if found_all[i] != 1:
#                     # if a neuron is not activated, it means that we have to move right
#                     if act_mask[i] != 1:
#                         b_min[i] =  current_b[i] 
#                     else:
#                         if len(strips[i]) == 0:
#                             # if we have no observations yet and active neurons, we  move left
#                             b_max[i] = current_b[i]
#                             obs_max = obs[i].detach().clone()
#                             observations_history[i][j].append((current_b[i], obs[i], 0))

#                         else:
#                             # if we have a single observation, we initialize the orthogonal basis
#                             if len(strips[i]) == 1:
#                                 orth_subspaces[i] = strips[i][-1][1].clone().detach().unsqueeze(0)
                            
#                             flag, norm = check_span(orth_subspaces[i], obs[i], norm_threshold=thresholds[i], debug=True)

#                             #NOTE: debug purposes
#                             print(f"current b:  {current_b[i].item()}  | b_min: {b_min[i].item()} | b_max: {b_max[i].item()} | Real position {b_sorted[i, j].item()}")
#                             if current_b[i] > b_sorted[i, j].item() and flag:
#                                 print("Warning: this observation should not be in the span, however it is")
#                                 print("Using per-sample computation...")
                                
#                                 _, obs[i] = check_real_weights(images, labels, model, i, debug=False)
#                                 flag, norm = check_span(orth_subspaces[i], obs[i], norm_threshold=thresholds[i], debug=True)
#                                 observations_history[i][j].append((current_b[i], obs[i], norm))

                                
#                                 # orth_comp = project_onto_orth_subspace(obs[i], orth_subspaces[i])
#                                 # lin_comb = torch.linalg.lstsq(orth_subspaces[i].T, obs[i].view(-1, 1)).solution
#                                 # # restore_image(lin_comb.T @ orth_subspaces[i], display=True, title='Linear comb')
#                                 # # restore_image(obs[i], display=True, title='observation')
#                                 # img_idx = torch.where(b_tensor[i] == b_sorted[i,j])
#                                 # restore_image(images[img_idx], display=True, title=f"Image {img_idx}")

                                
#                                 # for k in range(len(orth_subspaces[i])):
#                                 #     print(f"Dot product with previous comp. {k}: {torch.dot(orth_subspaces[i][k], orth_comp)}")
#                                 # weights_scaled, weights_not_scaled, loss_list, x_rec =  check_real_weights(images, labels, model, i, debug=False)
#                                 # w = torch.tensor([k[1] for k in weights_scaled]).double()
#                                 # act_idx = [i[0] for i in weights_scaled]
#                                 # obs_with_lin_comb = lin_comb.T @ orth_subspaces[i]
#                                 # print(f"residual: {torch.dot(orth_comp, orth_comp)}")
#                                 # b_test_max = torch.ones_like(b_max) * current_b[i] + 10

#                                 # test_obs = check_multiple_observations(b_max=b_test_max,
#                                 #                                         b = current_b.detach().clone(),
#                                 #                                         images=images,
#                                 #                                         labels=labels,
#                                 #                                         model=model,
#                                 #                                         criterion=criterion,
#                                 #                                         optimizer=optimizer,
#                                 #                                         direction=i,
#                                 #                                         epsilon=0.1, 
#                                 #                                         orth_subspace=orth_subspaces[i])
#                             if flag:
#                                 # if the new observation is in the span of the previous ones, we move right
#                                 b_min[i] = current_b[i].detach().clone()
                                
#                             else:
#                                 # if the new observation is not in the span of the previous one, we move left
#                                 b_max[i] = current_b[i].detach().clone()
#                                 obs_max = obs[i].detach().clone()

#                             observations_history[i][j].append((current_b[i].detach().clone(), obs[i].detach().clone(), norm))

#                             #NOTE: debug purposes
#                             # check_weights(images, labels, model, i)

                    
#                     interval[i] = b_max[i] - b_min[i]
#         # print(f"b_min: {b_min[i].item()} | b_max: {b_max[i].item()}")
#         b_min = b_max.detach().clone()
#         b_max = B_max.detach().clone()
#         interval = b_max-b_min

#     return strips, model.fc1.weight.data, model.fc1.bias.data


def main():

    set_seeds(3)

    images, all_labels = prepare_data()
    
    num_samples = 10
    sample = np.random.choice(range(images.shape[0]), size=num_samples, replace=False)
    images_client = images[sample]
    labels_client = all_labels[sample]

    n_directions = 10

    # STEP 1 + 2
    
    strips_dict, neurons, bias = find_strips(images=images_client,
                                                labels=labels_client,
                                                n_classes=10,
                                                n_directions=n_directions)
    
    # STEP 3: find corresponding strips

    corresponding_strips = find_corresponding_strips(strips_dict)   

    # STEP 4: reconstruct images
    recon_images = corresponding_strips[0][0].unsqueeze(0)

    # for k in corresponding_strips.keys():
    #     restore_image(corresponding_strips[k][0].unsqueeze(0), display=True)
    # restore_image(recon_images[0][0], display=True)
    

    
    show_true_images(images_client)

    for k in range(1, num_samples):
        # pick the k-the observation in direction 0
        obs = corresponding_strips[k][0]
        # pick the first k directions
        a = neurons[:k+1, :]
        # pick the associated kth-b coefficients
        b = bias[:k+1]
        coefficients, image = solve_linear_system(obs, a, b, recon_images)
        image = image / coefficients[-1]
        recon_images = torch.cat((recon_images, image.unsqueeze(0)), dim=0)
    
    print("Showing reconstructed images")
    for k in range(num_samples):
        restore_image(recon_images[k], display=True)



    # recon_images = reconstruct_images(corresponding_strips, images_client, n_directions)
if __name__ == '__main__':
    main()