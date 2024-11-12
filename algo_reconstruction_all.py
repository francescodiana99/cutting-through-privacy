import bisect
import random
import math
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def inside_span(observation, observations_points, observations):
    """
    Check if the observation is in the span of the images_reconstructed
    Args:
        observation: tensor of the observation to check
        observations_points: list of the b that is on the right of the recovered points (b_max of the associated observation)
        observations: dictionary containing history of the observations in the form of {b: observation}
    """
    observation = torch.unsqueeze(observation, 0)
    # Build matrix with checking observations
    for i in range(len(observations_points)):
        if i == 0:
            A = torch.unsqueeze(observations[observations_points[i]], 0)
        else:
            A = torch.cat((A,  torch.unsqueeze(observations[observations_points[i]], 0)))
    
    # Add other observations on the left to make sure that the rank of A is correct
    obs_check = [torch.unsqueeze(value, 0) for key, value in observations.items() if key<observations_points[-1]]
    random.shuffle(obs_check)
    for o in obs_check[:2*len(observations_points)]: # add additional 2 * len(observations_points) to make sure that the rank is correct (this might make it unstable)
        A = torch.cat((A, o))

    B = torch.cat((A, observation))
    rankA = torch.linalg.matrix_rank(A)
    rankB = torch.linalg.matrix_rank(B)
    if rankA<len(observations_points):
        print(f"****We have {len(observations_points)} points, however the rank A of size {A.shape[0]} is {rankA}")
    if rankA>len(observations_points):
        print(f"Another problem")

    #print(f"rankA: {rankA}, rankB: {rankB}")
    #If the two ranks are the same: careful double check
    if rankA == rankB:
        i = 0
        for o in obs_check[2*len(observations_points):]:
            A = torch.cat((A, o))
            B = torch.cat((B, o))
            new_rankA = torch.linalg.matrix_rank(A)
            new_rankB = torch.linalg.matrix_rank(B)
            if new_rankB > new_rankA:
                rankB = new_rankB
                break
            elif new_rankA > rankA:
                print(f"***Rethink rankA {rankA}/size {len(observations_points)} vs newrankA {new_rankA}/size {A.shape[0]} ")
            i += 1
            if i>2*(len(observations_points)): break
    return not (rankB>rankA)



def inside_span_artificial(observation, images_reconstructed, observations_points):
    """
    Check if the observation is in the span of the images_reconstructed, by using the true image locations
    Args:
        observation: tensor of the observation to check 
        images_reconstructed: tensor of the images that are already reconstructed
        observations_points: list of the b that is on the right of the recovered points
    """
    A = images_reconstructed
    observation = torch.unsqueeze(observation, 0)
    B = torch.cat((A, observation))
    rankA = torch.linalg.matrix_rank(A, rtol=0.00001)
    rankB = torch.linalg.matrix_rank(B, rtol=0.00001)

    if rankA<len(observations_points):
        print('Artificial case')
        print(f"****We have {len(observations_points)} points, however the rank A of size {A.shape[0]} is {rankA}")
    if rankA>len(observations_points):
        print('Artificial case')
        print(f"Another problem")
    return not (rankB>rankA)


def inside_span_test(observation, observations_points, observations):
    observation = torch.unsqueeze(observation, 0)
    # Build matrix with checking observations
    for i in range(len(observations_points)):
        if i == 0:
            A = torch.unsqueeze(observations[observations_points[i]], 0)
        else:
            A = torch.cat((A,  torch.unsqueeze(observations[observations_points[i]], 0)))

    A_before_scaled = A 
    rank_A_before_scaled = torch.linalg.matrix_rank(A_before_scaled, atol=0.00001)
    B_before_scaled = torch.cat((A_before_scaled, observation))
    rank_B_before_scaled = torch.linalg.matrix_rank(B_before_scaled, atol=0.00001)

    B = torch.cat((A, observation))

    # A_scaled = A * 1e3

    if rank_A_before_scaled != len(observations_points):
        print(f"***Warning Rank A is {rank_A_before_scaled},  however have {len(observations_points)} points")
        print(f"Ranks with scaled A before adding obs: {rank_A_before_scaled}")
        # u, s, vh = svd(A)
        # print("Singular values of A before adding obs: ", s_before)
        # print("Condition number of A before adding obs: ", s_before[0]/s_before[-1])
        # print("Singular values of A after adding obs: ", s)

    if rank_A_before_scaled<len(observations_points):
        print(f"****We have {len(observations_points)} points, however the rank A of size {A_before_scaled.shape[0]} is {rank_A_before_scaled}")
    if rank_A_before_scaled>len(observations_points):
        print(f"Another problem")
        print(f"A rank is {rank_A_before_scaled}, however we have {len(observations_points)} points")

    return not (rank_B_before_scaled>rank_A_before_scaled)


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

    observations_points = [] # list of the b that is on the right of the recovered points (current b_max of the iteration))
    recovered_points = []


    for j in range(images.shape[0]):
        # print("---------------------------------------------")
        print(f"Round {j} : b_min {b_min}, b_max {b_max}")

        if len(observations_points) != j:
            print('-Debug issue-found_all------')
            exit(0)
        while interval>min(epsilon_list)/2: # the minimum value is the minimum distance between two points

            hyperplane = b_min + (b_max-b_min)/2
            indices_left = bisect.bisect_left(b_sorted.numpy(), hyperplane)
            images_indices_left = indices[:indices_left] # images that are on the left of the hyperplane

            if len(images_indices_left) == 0:
                b_min = hyperplane
            else:
                # TODO: we should use the real weights
                # generating random weithts for the observation
                alpha = np.random.uniform(-1,1, size=images_indices_left.shape[0])
                alpha = torch.tensor(alpha/sum(alpha), dtype=torch.float32)
                # using a fake observation, based on the linear combination of the images on the left
                observation = alpha @ images[images_indices_left] 
                observations[hyperplane] = observation


                # checking the observations to decide which direction to go
                if len(observations_points ) == 0:
                    b_max = hyperplane
                
                else:
                    # flag = inside_span_test(observation, observations_points, observations)
                    # flag = inside_span(observation, observations_points, observations)

                    # TODO: this should not use the artificial computation
                    flag = inside_span_artificial(observation, images_reconstruct, observations_points)
                    if flag:
                        b_min = hyperplane
                    else:
                        b_max = hyperplane
                        # additional checks
                        if len(images_indices_left) == len(observations_points) and not(flag):
                            print("---Warning:Should be the same rank, however different rank")
                        if len(images_indices_left) > len(observations_points) and flag:
                            print(f"---Warning:Should be the different rank {len(images_indices_left)}, however same rank {len(observations_points)}")
            
            interval = b_max - b_min

        images_reconstruct = images[indices[:j+1]]

        if b_max in observations.keys():
            observations_points.append(b_max) #Change later: should be the hyperplan cloest to b_max in observations{key}
            print(f"Observation point is in {b_max} and true point is in {b_sorted[j]}")

        else:
            print("Details error to reconsider")
        recovered_points.append(b_min + (b_max-b_min)/2)
        print("Last cut: ", recovered_points[-1])

        # DEBUG
        # if j > 0:
        #     if torch.all(observation == observations_images[0]):
        #         print(f"---Warning: The same observation is added again")
        #         print(f"B_min {b_min}, B_max {b_max}")
        #     else: 
        #         if len(images_indices_left) == 0:
        #             print(f"---Warning: No images on the left of the hyperplane")
        #             print(f"b_min before the last step: {old_b_min}")
        #             print("True B cuts list: ", b_sorted)
        #             print("B_min list: ", b_min_list)
        #             print("B_max list: ", b_max_list)
        #             hyperplane_debug = recovered_points[-1]
        #             indices_left_debug = bisect.bisect_left(b_sorted.numpy(), hyperplane_debug)
        #             images_indices_left_debug = indices[:indices_left_debug]
        #             print(f"images_indices_left_debug: {images_indices_left_debug}")
        #             observation_debug = alpha @ images[images_indices_left_debug]
        #             print(f"Observation debug: {observation_debug}")
        #             print(f"Observation: {observation}")
        #             print(f"Interval: {interval}")
        #             print(f"Epsilon: {min(epsilon_list)/2}")
        # alphas.append(alpha)
        # print(images_indices_left)
        # END DEBUG

        print(f"Found a point in {b_min + (b_max-b_min)/2} and true point is in {b_sorted[j]}, b_max for oberservation {b_max}")
        # hyperplane_last = recovered_points[-1]
        # indices_left_last = bisect.bisect_left(b_sorted.numpy(), hyperplane_last)
        # images_indices_left_last = indices[:indices_left_last]
        # observation_last = alpha @ images[images_indices_left_last]

        # if len(recovered_points) == 1: 
        #     observations_images = torch.unsqueeze(observation_last, 0)
        # else:
        #     observations_images = torch.cat((observations_images, torch.unsqueeze(observation_last, 0)))
        # if np.linalg.matrix_rank(observations_images.numpy()) != j+1:
        #     print(f"Rank of the observation matrix is {np.linalg.matrix_rank(observations_images.numpy())} and should be {j+1}")
            # print(observations_images)
        #Reintialize the start b_min, b_max by looking at the observations
        b_min = b_max
        b_max = B_max
        interval = b_max-b_min
        # TODO: currently using the real position, later switch to observations.
        real_obs = []
        for im_left in range(len(images)):
            alpha = np.random.uniform(-1,1, size=images[:im_left+1].shape[0])
            alpha = torch.tensor(alpha/sum(alpha), dtype=torch.float32)
            # using a fake observation, based on the linear combination of the images on the left
            observation = alpha @ images[:im_left+1] 
            real_obs.append(observation)
        real_obs = torch.stack(real_obs)
        # shuffle the rows to not have all the images observation in the same order
        real_obs = shuffle_rows(real_obs)
    return recovered_points, a, observations, real_obs


def shuffle_rows(matrix):
    """
    Shuffle the rows of the matrix
    Args:
        matrix: tensor of the matrix to shuffle
    """
    indices = torch.randperm(matrix.shape[0])
    return matrix[indices]


def find_same_image(v, u):
    for j in range(1, u.shape[0]):
        print("j = ", j)
        q, r = np.linalg.qr(u[:j])
        assert (np.linalg.matrix_rank(q)) == (j), f"The matrix should be full rank but has rank {np.linalg.matrix_rank(q)}"

        # case when i>1
        if v.shape[0] != 1:
            v_prev = v[:v.shape[0]-1, :]
            v_i = v[-1, :]

            # find the orthogonal subspace to the subspace spanned by u_1, ..., u_{j-1}, v_1, ..., v_{i-1}
            subspace = torch.cat((v_prev, u[:j, :]), 0)
            A = torch.tensor(sp.linalg.null_space(subspace.numpy()))
        # case when i=1
        else:
            v_i = v.squeeze(0)
            A = torch.tensor(sp.linalg.null_space(u[:j, :].numpy()))

        p_v = (A @ torch.inverse(A.T @ A) @ A.T @ v_i.T).unsqueeze(1)
        p_u = (A @ torch.inverse(A.T @ A) @ A.T @ u[j, :].T).unsqueeze(1)
        proj_matrix = torch.cat((p_v, p_u), 1)
        if torch.linalg.matrix_rank(proj_matrix) == 1:
            print("Found same image!!")
            return u[j, :]
        
    print("Error: No corresponding image found")
    return torch.zeros_like(u[j, :])



        

def find_corresponding_strips(strips_dict, observation_idx, v_obs_list, corresponding_strips):
    v = v_obs_list[:observation_idx + 1] #  strips corresponding to all the observations up to v_i
    q, r = np.linalg.qr(v)
    assert (np.linalg.matrix_rank(q)) == v.shape[0], f"The matrix should be full rank but has rank {np.linalg.matrix_rank(q)}"
    # for each pair of observation u_i, v_j check if they have the same projection onto the subspace orthogonal to the one spanned by the previous observations
    print("i = ", observation_idx + 1)
    for key, _ in strips_dict.items():
        if key != 0:
            # for each strip in a specific fixed direction, find the one observing the same image as last observation
            obs = find_same_image(v, strips_dict[key])
            corresponding_strips[observation_idx].append(obs)
    return corresponding_strips


def solve_linear_system(observation, directions, b, images):
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

    A_images = torch.cat((torch.transpose(images, 0,1), torch.zeros(k, k-1)), dim=0)
    A_b = torch.transpose(torch.cat((torch.zeros(d), - b), dim=0).unsqueeze(0), 0, 1)
    A_a = torch.cat((torch.eye(d), directions), dim=0)
    A = torch.cat((A_images, A_b, A_a), dim=1)

    B = torch.cat((observation, torch.zeros(k)), dim=0).double()

    x = torch.linalg.solve(A, B)

    coefficients = x[k:]
    image = x[:k]


    return coefficients, image


    


        
def main():

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 1024

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        if i == 0:
            images = torch.flatten(inputs, start_dim=1)
        else:
            images = torch.cat((images, torch.flatten(inputs, start_dim=1)))

    num_samples = 4
    sample = np.random.choice(range(images.shape[0]), size=num_samples, replace=False)
    images_client = images[sample]

    strips_dict = {} # dictionary in the form {"direction": [v_1, v_2, ...]}, contains real point observations.
    directions = {} # dictionary in the form {"direction": [a_1, a_2, ...]}
    observations_dict = {} # dictionary in the form {"direction": [observation_1, observation_2, ...]}
    strips_b_dict = {} # dictionary in the form {"direction": [b_1, b_2, ...]}
    n_directions = 4
    # STEP 1 + 2: choose n random diretions and for each direction find a strip where the image lies.
    for i in range(n_directions):
            strips_b_dict[i], directions[i], observations_dict[i], strips_dict[i] = find_strips(images_client)

    v_obs_list = strips_dict[0]
    corresponding_strips = {i: [v_obs_list[i]] for i in range(num_samples)}

    # STEP 3: for each strip, find n-1 strips that correspond to the same observation
    for i in range(num_samples): # for each observation find the corresponding strips 
        corresponding_strips = find_corresponding_strips(strips_dict, i, v_obs_list, corresponding_strips)

    # STEP 4: solve n linear systems to find the images
    recon_images = images_client[:1].clone().detach() # list of the reconstructed images (starting with the first two, but this should be recovered)
    for k in range(1, num_samples):
        
        # pick the k-the observation in direction 0
        obs = corresponding_strips[k][0]
        # pick the first k directions
        a = torch.stack([torch.tensor(directions[i]) for i in range(k+1)])
        # pick the associated kth-b coefficients
        b = torch.tensor([strips_b_dict[i][k] for i in range(k+1)])
        image, coefficients = solve_linear_system(obs, a, b, recon_images)
        recon_images = torch.cat((recon_images, image.unsqueeze(0)), dim=0)

        # STEP 5: compare the reconstructed image with the true image
        true_image = images_client[k].view(3, 32, 32)
        recon_image = image.view(3, 32, 32)

        true_image = true_image.permute(1, 2, 0)
        recon_image = recon_image.permute(1, 2, 0)

        # unnormalize and restore channels
        std = torch.tensor((0.5, 0.5, 0.5))
        mean = torch.tensor((0.5, 0.5, 0.5))

        true_image = true_image * std + mean
        recon_image = recon_image * std + mean

        # plot the images
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(true_image)
        axs[0].axis('off')
        axs[1].imshow(recon_image)
        axs[1].axis('off')
        plt.show()


if __name__ == "__main__":
    main()
