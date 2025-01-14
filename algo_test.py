import bisect
import random
import math
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from utils_search import *
from utils_misc import *
from utils_test import *
from attacks.utils import prepare_resnet

def projection(x, y):
    """
    Project x onto y
    """
    return (torch.dot(x, y)/torch.dot(y, y))*y


def project_onto_orth_subspace(x, orth_subspace):
    residual = x
    
    for i in range(orth_subspace.shape[0]):
        residual -= projection(x, orth_subspace[i])
    return residual

def find_strips(images):
    image_size = images.shape[1]

    a = np.random.rand(image_size)
    bounderay_image_min = -1*np.ones(image_size)
    b_1 = -np.dot(a, bounderay_image_min)
    bounderay_image_max = 1*np.ones(image_size)
    b_2 = -np.dot(a, bounderay_image_max)

    b_min = min(b_1, b_2)
    b_max = max(b_1, b_2)

    b_list = []
    for i in range(images.shape[0]):
        b_list.append(-np.dot(a, images[i]))

    b_sorted, indices = torch.sort(torch.tensor(b_list))
    epsilon_list = abs(b_sorted[:-1]-b_sorted[1:])
    B_max = b_max
    interval = b_max-b_min
    observations = dict()

    observation_points = []
    recovered_points = []

    for j in range(images.shape[0]):
        print(f"Round {j} : b_min {b_min}, b_max {b_max}")

        while interval>min(epsilon_list)/2:
            current_b = b_min + (b_max-b_min)/2 # find the middle point
            indices_left = bisect.bisect_left(b_sorted.numpy(), current_b)
            images_indices_left = indices[:indices_left]
            if len(images_indices_left) == 0:
                b_min = current_b
            else:
                # TODO: we should use the real weights
                # generating random weithts for the observation
                alpha = np.random.uniform(-1,1, size=images_indices_left.shape[0]) 
                alpha = torch.tensor(alpha/sum(alpha), dtype=torch.float32)
                # using a fake observation, based on the linear combination of the images on the left
                observation = alpha @ images[images_indices_left] 
                observations[current_b] = observation


                # checking the observations to decide which direction to go
                if len(observation_points) == 0: 
                    b_max = current_b
                
                else:
                    # flag = inside_span_test(observation, observations_points, observations)
                    # flag = inside_span(observation, observations_points, observations)

                    # TODO: this should not use the artificial computation
                    if len(images_reconstruct) == 1:
                        orth_subspace = images_reconstruct[0].unsqueeze(0)

                    # observation = 0.3 * orth_subspace[0] 

                    flag, residual, norm = check_span_artificial(observation, images_reconstruct, observation_points, orth_subspace)
                    if flag:
                        b_min = current_b
                    else:
                        b_max = current_b
                        # additional checks
                        if len(images_indices_left) == len(observation_points) and not(flag):
                            print("---Warning: an image has been detected even though it should not")
                            print(f"Norm of the residual: {torch.dot(residual, residual)}")
                            orth_subspace = orth_subspace.double()
                            for k in range(orth_subspace.shape[0]):
                                residual = residual.double()
                                print(f"Dot product with the orthogonal component {k}: {torch.dot(residual, orth_subspace[k])}")
                            print(f"current  b:  {current_b}")
                            print(f"Next image is observed at b: {b_sorted[j]}")
                        if len(images_indices_left) > len(observation_points) and flag:
                            print(f"---Warning: An image has not been detected even though it should")
                            print(f"current  b:  {current_b}")
                            print(f"Norm of the residual: {torch.dot(residual, residual)}")

            interval = b_max - b_min

        images_reconstruct = images[indices[:j+1]]

        if b_max in observations.keys():
            observation_points.append(b_max) #Change later: should be the hyperplan cloest to b_max in observations{key}
            print(f"Observation point is in {b_max} and true point is in {b_sorted[j]}")

        else:
            print("Details error to reconsider")
        recovered_points.append(b_min + (b_max-b_min)/2)
        if len(recovered_points) > 1:
                #TODO: adding a fake observation that exactly cut the space in the image, this should be replaced by the real observation
                obs_idx = bisect.bisect_left(b_sorted.numpy(), (b_min + (b_max-b_min)/2))
                obs_images = indices[:obs_idx]
                alpha = np.random.uniform(-1,1, size=obs_images.shape[0])
                alpha = torch.tensor(alpha/sum(alpha), dtype=torch.float32)
                observation = alpha @ images[obs_images]
                observation = observation.double()
                orth_subspace = orth_subspace.double()
                add_orth_comp = project_onto_orth_subspace(observation, orth_subspace)
                print(f"added orthogonal component. Dot prodct with the previous orthogonal components: {torch.dot(add_orth_comp, orth_subspace[-1])}")
                orth_subspace = torch.cat((orth_subspace, add_orth_comp.unsqueeze(0)), 0)
        print(f"Found a point in {b_min + (b_max-b_min)/2} and true point is in {b_sorted[j]}, b_max for oberservation {b_max}")

        b_min = b_max
        b_max = B_max
        interval = b_max-b_min


def test_step_1():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 1024

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        if i == 0:
            images = torch.flatten(inputs, start_dim=1)
        else:
            images = torch.cat((images, torch.flatten(inputs, start_dim=1)))

    num_samples = 100
    sample = np.random.choice(range(images.shape[0]), size=num_samples, replace=False)
    images_client = images[sample]
    n_directions = 8

    strips_dict = {} # dictionary in the form {"direction": [v_1, v_2, ...]}, contains real point observations.
    directions = {} # dictionary in the form {"direction": [a_1, a_2, ...]}
    observations_dict = {} # dictionary in the form {"direction": [observation_1, observation_2, ...]}
    strips_b_dict = {} # dictionary in the form {"direction": [b_1, b_2, ...]}
    # STEP 1 + 2: choose n random diretions and for each direction find a strip where the image lies.
    for i in range(n_directions):
            strips_b_dict[i], directions[i], observations_dict[i], strips_dict[i] = find_strips(images_client)

    
def test_projection():
    random_matrix = torch.rand(1000, 3074)
    matrix_rank = torch.linalg.matrix_rank(random_matrix)
    print(f"Matrix rank: {matrix_rank}")
    print(f" Subspace rank: {torch.linalg.matrix_rank(random_matrix[1:])}")
    orth_subspace = random_matrix[0].unsqueeze(0)
    for i in range(1, random_matrix.shape[0]):
        residual = project_onto_orth_subspace(random_matrix[i], orth_subspace)
        orth_subspace = torch.cat((orth_subspace, residual.unsqueeze(0)), 0)
        print(f"Orthogonal subspace rank: {torch.linalg.matrix_rank(orth_subspace)}")
        print(f"Norm of the residual: {torch.dot(residual, residual)}")
        print(f"Dot product with all the orthogonal component for residual {i}")
        for j in range(orth_subspace.shape[0]):
            print(f"{j}: {torch.dot(residual, orth_subspace[j])}")


    # random_vector = torch.rand(10)
    # lin_dep = 0.2 * random_vector
    # lin_ind =  torch.rand(10)
    # print(f"Rank of the subspace ind: {torch.linalg.matrix_rank(torch.stack((random_vector, lin_ind), dim=0))}")
    # print(f"Rank of the subspace dep: {torch.linalg.matrix_rank(torch.stack((random_vector, lin_dep), dim=0))}")

    # proj_dep = projection(lin_dep, random_vector)
    # ort_dep = lin_dep - proj_dep
    
    # print(f"Projection dep scalar factor: {proj_dep/ random_vector}")
    # print(f"Ort dep norm: {torch.dot(ort_dep, ort_dep)}")

    # proj_ind = projection(lin_ind, random_vector)
    # ort_ind = lin_ind - proj_ind
    
    # print(f"Projection ind scalar factor: {proj_ind / random_vector}")
    # print(f"Ort ind norm: {torch.dot(ort_ind, ort_ind)}")
    # print(f"dot product dep:  {torch.dot(ort_dep, random_vector)}")
    # print(f"dot product ind:  {torch.dot(ort_ind, random_vector)}")


def shuffle_images(images):
    """
    Shuffle the images
    """
    indices = np.random.permutation(images.shape[0])
    return images[indices], indices

def test_step_3():
    """
    Step 3: For each strip, find n-1 strips that correspond to the same image
    This method tests the second step of the algorithm using random matrices.
    """

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 1024

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        if i == 0:
            images = torch.flatten(inputs, start_dim=1)
        else:
            images = torch.cat((images, torch.flatten(inputs, start_dim=1)))

    num_samples = 100
    sample = np.random.choice(range(images.shape[0]), size=num_samples, replace=False)
    images_client = images[sample]
    n_directions = 100

    strip_dict = {}
    image_idx_dict = {}
    
    for d in range(n_directions):
        # shuffle the images to get a random order for each direction
        images_client, indices = shuffle_images(images_client)
        image_idx_dict[d] = indices
        for j in range(images_client.shape[0]):
            real_obs = []
            for im_left in range(len(images_client)):
                alpha = np.random.uniform(-1,1, size=images_client[:im_left+1].shape[0])
                alpha = torch.tensor(alpha/sum(alpha), dtype=torch.float32)
                # using a fake observation, based on the linear combination of the images on the left
                observation = alpha @ images_client[:im_left+1] 
                real_obs.append(observation)
            real_obs = torch.stack(real_obs)
        strip_dict[d] = real_obs

    # check if the observation corresponds to the same image
    v = strip_dict[0] # fix direction v
    v_orth = v[0].unsqueeze(0) # first element of the orthogonal supspace spanned by v
    strip_dict.pop(0)

    for i in range(v.shape[0]):
        if i != 0:
            residual = project_onto_orth_subspace(v[i], v_orth)
            v_orth = torch.cat((v_orth, residual.unsqueeze(0)), 0)

    corresponding_obs = {i: v[i].unsqueeze(0) for i in range(v.shape[0])}
    for d in strip_dict.keys():
        u = strip_dict[d]
        for i in range(v_orth.shape[0]):
            curr_orth = v_orth[:i +1] # select the first i orthogonal components in v_orth

            not_found = True
            j = 0

            while not_found:
                # project the observation u[j] onto the orthogonal subspace spanned by the first i orthogonal components of v
                residual = project_onto_orth_subspace(u[j], curr_orth)
                # if the residual is zero, then the observation u[j] is in the span of the orthogonal subspace
                if torch.dot(residual, residual) < 1e-5:
                    # add the observation to the corresponding observation list
                    corresponding_obs[i] = torch.cat((corresponding_obs[i], u[j].unsqueeze(0)), 0)
                    not_found = False
                    u = torch.cat((u[:j], u[j+1:]), 0)
                    print("Correspondence found")
                    image_idx_dict[d] = np.delete(image_idx_dict[d], j)


                else:
                    # add the residual to the orthogonal subspace
                    curr_orth = torch.cat((curr_orth, residual.unsqueeze(0)), 0)
                    j += 1
                    if j == u.shape[0]:
                        not_found = False
                        print(f"Warning: an image correspondence has not been detected even though it should")
                        for i in range(curr_orth.shape[0]):
                            print(f"Dot product with the orthogonal component {i}: {torch.dot(residual, curr_orth[i])}")
                        print('ok')


def test_step_4():

    images, all_labels = prepare_data()
    num_samples = 4
    sample = np.random.choice(range(images.shape[0]), size=num_samples, replace=False)
    images_client = images[sample]
    labels_client = all_labels[sample]

    n_directions = 4
    image_size = images_client.shape[1]

    model = NN(input_dim=image_size, n_classes=10, hidden_dim=n_directions)
    model = model.double()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    b_tensor = - torch.matmul(model.fc1.weight, torch.transpose(images_client, 0, 1))

    # useful for checking how good is the binary search
    # b_sorted, indices = torch.sort(torch.tensor(b_tensor.detach().clone()), dim=1)

    corresponding_strips = dict()

    for i in range(images_client.shape[0]):
        obs, dL_dA, dL_db = get_observations(images=images_client,
                                                labels=labels_client,
                                                model=model,
                                                criterion=criterion,
                                                optimizer=optimizer,
                                                b=b_tensor[:, i] 
                                                )
        
        strips = [obs[i] for i in range(obs.shape[0])]
        corresponding_strips[i] = strips

    

    for k in range(1, num_samples):
        # pick the k-the observation in direction 0
        obs = corresponding_strips[k][0]
        # pick the first k directions
        a = model.fc1.weight.data[:k+1, :].detach()
        # pick the associated kth-b coefficients
        b = b_tensor[:k+1]
        coefficients, image = solve_linear_system(obs, a, b, recon_images)
        image = image / coefficients[-1]
        recon_images = torch.cat((recon_images, image.unsqueeze(0)), dim=0)


def plot_weights():
    images, all_labels = prepare_data()
    num_samples = 2
    sample = np.random.choice(range(images.shape[0]), size=num_samples, replace=False)
    images_client = images[sample]
    labels_client = all_labels[sample]

    n_directions = 1
    image_size = images_client.shape[1]

    model = NN(input_dim=image_size, n_classes=10, hidden_dim=n_directions)
    model.fc2.bias.data = torch.tensor([10.0] * 10)
    model = model.double()

    print(model.parameters())

    b_tensor = - torch.matmul(model.fc1.weight, torch.transpose(images_client, 0, 1))
    b_sorted, indices = torch.sort(torch.tensor(b_tensor.detach().clone()), dim=1)
    epsilon = torch.tensor([0.01] * b_tensor.shape[0])


    # NOTE: Check this. Now using a version that is artificial
    # b_min = torch.min(b_sorted, dim=1).values  
    # b_min = b_sorted[1]
    b_max = torch.max(b_sorted, dim=1).values  
    epsilon = 1

    current_b = b_max.detach().clone()  + 0.1
    x = []
    y = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []
    y6 = []
    while current_b[0] <= b_max[0] + 10:
        model.fc1.bias.data = current_b.detach().clone()
        weights_scaled_100, _ , loss_list_100, _, _, _ = check_real_weights(images=images_client,
                                    labels=labels_client,
                                    model=model,
                                    direction=0,
                                    scale_factor = 1000,
                                    debug=False)
        weights_scaled, weights_not_scaled, loss_list, _, _, _ = check_real_weights(images=images_client,
                                    labels=labels_client,
                                    model=model,
                                    direction=0,
                                    scale_factor = 10,
                                    debug=False)
        weights_scaled_no_temp, weights_not_scaled_no_temp, loss_list_no_temp, _, _, _ = check_real_weights(images=images_client,
                                    labels=labels_client,
                                    model=model,
                                    direction=0,
                                    scale_factor = 1,
                                    debug=True)
        x.append(current_b[0].detach().clone())
        y.append(weights_scaled[1][1])
        y2.append(loss_list[1])
        y3.append(weights_scaled_no_temp[1][1])
        y4.append(loss_list_no_temp[1])
        y5.append(weights_scaled_100[1][1])
        y6.append(loss_list_100[1])
        
        current_b[0] += epsilon

    restore_image(images_client[1], display=False)

    fig, ax = plt.subplots(1, 3)
    ax[1].plot(x, y, label='weight')
    ax[1].plot(x, y2, label='loss')
    # b_list = b_sorted.tolist()
    # ax.axvline(x=b_list)
    ax[1].set(xlabel='b', title='Alpha 2 with T=10')
    ax[1].set_ylim([-5,5])
    print(y)

    ax[0].plot(x, y3, label='weight')
    ax[0].plot(x, y4, label='loss')
    ax[0].set(xlabel='b', ylabel='dL/db', title='Alpha 1 with no scaling')
    ax[0].set_ylim([-5,5])

    ax[2].plot(x, y5, label='weight')
    ax[2].plot(x, y6, label='loss')
    # b_list = b_sorted.tolist()
    # ax.axvline(x=b_list)
    ax[2].set(xlabel='b', title='Alpha 2 with T=1000')
    ax[2].set_ylim([-5,5])
    print(y5)

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()
    

def test_layer_modification():
    images, all_labels = prepare_data()
    num_samples = 2
    sample = np.random.choice(range(images.shape[0]), size=num_samples, replace=False)
    images_client = images[sample]
    labels_client = all_labels[sample]

    n_directions = 3
    n_control_neurons = 1
    image_size = images_client.shape[1]

    model = NN(input_dim=image_size, n_classes=10, hidden_dim=n_directions+n_control_neurons)
    model = model.double()

    
    b_tensor = - torch.matmul(model.fc1.weight, torch.transpose(images_client, 0, 1))
    b_sorted, indices = torch.sort(torch.tensor(b_tensor.detach().clone()), dim=1)
    epsilon = torch.tensor([0.01] * b_tensor.shape[0])

    model.fc1.bias.data[n_directions:] = 1e8
    model.fc1.bias.data[:n_directions] = b_sorted[:n_directions, -1] + 0.1
    model.fc2.bias.data = torch.tensor([1.] * 10).double()
    model.fc2.weight.data = (torch.ones_like(model.fc2.weight) * 0.1).double()

    for i in range(n_directions):


        weights_scaled, weights_activated, loss_list = check_real_weights(images=images_client, 
                                                                    labels=labels_client,
                                                                    model=model,
                                                                    direction=i,
                                                                    )
        print(weights_scaled)


def test_one_direction_attack():
    args = parse_args()

    # set_seeds(3)

    images, all_labels = prepare_data()

    num_samples = 500
    sample = np.random.choice(range(images.shape[0]), size=num_samples, replace=False)
    images_client = images[sample]
    labels_client = all_labels[sample]

    strips_obs, strips_b, neurons, bias, dL_db_history = find_strips_parallel(images=images_client,
                                                labels=labels_client,
                                                n_classes=10,
                                                n_directions=1,
                                                classification_bias_value=1e6,
                                                classification_weights_value=1e-3,
                                                control_bias=0,
                                                noise_norm=0,
                                                epsilon=1e-5,
                                                threshold=1e-6,
                                                directions_weights_value=1e-3,
                                                device='cuda')
    
    print("First_step_ok")

    # show_true_images(images_client)
    
    images_reconstructed = [strips_obs[0, 0]]
    print(get_ssim(images_client[0], strips_obs[0, 0]))
    restore_images([images_reconstructed[0], images_client[0]], device=args.device, display=True, title="Reconstructed image 0")
    dL_db_list = [dL_db_history[0, 0].item()]
    for k in range(1, num_samples):
        obs = strips_obs[k, 0]
        dL_db_k = (dL_db_history[k, 0] - dL_db_history[k-1, 0]).item()
        dL_db_list.append(dL_db_k)
        alphas = [i/dL_db_history[k, 0].item() for i in dL_db_list]
        print(f"Alphas for round {k}: {alphas}")
        recon_image = (obs - sum([alphas[i] * images_reconstructed[i] for i in range(len(images_reconstructed))]))/ alphas[-1]
        # restore_images([images_reconstructed[0], recon_image], device=args.device, display=True, title="Reconstructed image 0")

        # recon_image = (k + 1) * (obs - (sum(images_reconstructed)/ (k+1)))
        images_reconstructed.append(recon_image)
    
    paired_images = couple_images(images_reconstructed, images_client)
    for k in range(num_samples):
        ssim = get_ssim(paired_images[k][0], paired_images[k][1])
        print(f"SSIM for image {k}: {ssim}")
        # restore_images([paired_images[k][0], paired_images[k][1]], device=args.device, display=True, title=f"Reconstructed image {k}| SSIM: {ssim}")

def couple_data(reconstructed_data, real_data):
    """Pair each tensor with the one with the minimum norm difference"""
    paired_data = []
    for i in range(len(reconstructed_data)):
        min_norm = 1e10
        for j in range(len(real_data)):
            norm = torch.norm(reconstructed_data[i] - real_data[j])
            if norm < min_norm:
                min_norm = norm
                min_idx = j
        paired_data.append((reconstructed_data[i], real_data[min_idx], min_norm))
    return paired_data


def test_parallel_attack():
    # args = parse_args()
    dataset_name = 'cifar10'
    num_samples = 10
    images, all_labels, n_classes = prepare_data(double_precision=True, dataset=dataset_name,n_samples=num_samples, model_type='fc')


    sample = np.random.choice(range(images.shape[0]), size=num_samples, replace=False)
    images_client = images[sample]
    labels_client = all_labels[sample]
    strips_obs, dL_db_history, corr_idx = find_observations(images_client, labels_client, n_classes=n_classes, control_bias=1e13, hidden_layers=[100], 
                      input_weights_scale=1e-9, classification_weight_scale=1e-3, device='cuda', epsilon=1e-12, obs_atol=1e-5, obs_rtol=1e-6)
    # strips_obs, dL_db_history, corr_idx = find_observations_cnn(images_client, labels_client, n_classes=n_classes, dataset_name=dataset_name, control_bias=1e12, n_neurons=100, 
    #                   weights_scale=1e-5, classification_weight_scale=1e-3, device='cuda', epsilon=1e-6, obs_atol=1e-5, obs_rtol=1e-6)
    images_reconstructed = [strips_obs[0]]
    # restore_images([images_reconstructed[0], images_client[0]], device=args.device, display=True, title="Reconstructed image 0")

    dL_db_list = [dL_db_history[0]]

    for k in range(1, len(dL_db_history)):
        obs = strips_obs[k]
        dL_db_k = dL_db_history[k] - dL_db_history[k-1]
        dL_db_list.append(dL_db_k)
        alphas = [i/dL_db_history[k] for i in dL_db_list]
        # print(f"Alphas for round {k}: {alphas}")
        recon_image = (obs - sum([alphas[i] * images_reconstructed[i] for i in range(len(images_reconstructed))]))/ alphas[-1]
        # restore_images([images_reconstructed[0], recon_image], device=args.device, display=True, title="Reconstructed image 0")

        # recon_image = (k + 1) * (obs - (sum(images_reconstructed)/ (k+1)))
        images_reconstructed.append(recon_image)

    if dataset_name != 'adult':
        images_client = images_client.flatten(start_dim=1)
        # paired_images = couple_images(images_reconstructed, images_client)
        paired_images = [(images_reconstructed[i], images_client[corr_idx[i]]) for i in range(len(images_reconstructed))]
        count = 0
        for k in range(len(paired_images)):
            ssim = get_ssim(paired_images[k][0], paired_images[k][1], dataset_name=dataset_name)
            if ssim > 0.9:
                count += 1
            print(f"SSIM for image {k}: {ssim}")
        
        restore_images([paired_images[-1][0], paired_images[-1][1]], device='cuda', display=True, title="Last Reconstructed image", dataset_name=dataset_name)
        print(f"Number of images with SSIM > 0.9: {count}")
        
    else:
        paired_samples = couple_data(images_reconstructed, images_client)
        avg_norm_diff = sum([i[2] for i in paired_samples])/len(paired_samples)
        max_norm_diff = max([i[2] for i in paired_samples])
        min_norm_diff = min([i[2] for i in paired_samples])

        print(f"Average norm difference: {avg_norm_diff}")
        print(f"Max norm difference: {max_norm_diff}")
        print(f"Min norm difference: {min_norm_diff}")

    abs_alphas = [abs(i) for i in alphas]
    print(alphas)
    print(f"10 smallest alphas: {sorted(abs_alphas)[:10]}")
    print(f"10 largest alphas: {sorted(abs_alphas)[-10:]}")

    print(sum(alphas))
    

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

def test_resnet_forward():
    model = prepare_resnet('resnet18', 'imagenet', 1000, 1e-3, 1e12, 1e-3, 1e12)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y)
def main():

    set_seeds(42)


    # test_projection()

    # test_step_1()
    # test_step_3()
    # step_4()
    # test_step_4()
    # test_one_direction_attack()
    # test_layer_modification()
    # test_image_isolation(1024, n_trials=10000, var=10)

    test_parallel_attack()
    # test_resnet_forward()
    # pred_list = check_predictions()
    # print(pred_list)


def test_image_isolation(n_samples=30000, n_trials=10, var=1): 
    images, all_labels = prepare_data() 
    images = images.to('cuda').float()
    rnd_idx = np.random.choice(images.shape[0], size=n_samples, replace=False)
    images = images[rnd_idx].to('cuda')
    labels = all_labels[rnd_idx].to('cuda')
    for i in range(n_trials):
        A = (2 * torch.rand(10000, images.shape[1]) - 1).to('cuda')
        # A = torch.normal(0, var, size=(10000, images.shape[1])).to('cuda')
        b_tensor = - torch.matmul(A, torch.transpose(images, 0, 1))
        min_idx = torch.argmin(b_tensor, dim=1)
        if i == 0:
            act_hist = min_idx.clone()
        else:
            act_hist = torch.cat((act_hist, min_idx), dim=0)
    # print(b_tensor)
    # print(b_tensor.shape)
    uniques, counts = torch.unique(act_hist, return_counts=True)
    labels_act = labels[uniques]
    unique_labels, count_labels = torch.unique(labels_act, return_counts=True)
    print(f"Number of isolated images: {uniques.shape[0]}")
    print(f"Most isolated image: {torch.argmax(counts)} | Number of isolations: {torch.max(counts)}")
    print(f"Least isolated image: {torch.argmin(counts)} | Number of isolations: {torch.min(counts)}")
    print(f"Avg number of isolations: {torch.sum(counts)/n_samples}")
    print(f"Number of unique labels: {unique_labels.shape[0]}")
    print(f"Most isolated label: {torch.argmax(count_labels)} | Number of isolations: {torch.max(count_labels)}")
    print(f"Least isolated label: {torch.argmin(count_labels)} | Number of isolations: {torch.min(count_labels)}")
    plt.hist(act_hist.cpu().detach().numpy(), bins=n_samples)
    plt.show()
    plt.hist(labels_act.cpu(), bins=unique_labels.shape[0])
    plt.show()

if __name__ == "__main__":
    main()
