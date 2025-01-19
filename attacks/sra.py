from collections import defaultdict
import copy
import os
import torch
from torch import nn
from utils_misc import prepare_data
from abc import ABC, abstractmethod
import numpy as np
from utils_test import couple_inputs, get_psnr, get_ssim
import logging
import math 
class BaseSampleReconstructionAttack(ABC):
    """
    Class representing a sample reconstruction attack in federated learning.
    This attack aims to reconstruct the training samples of a target client in federated learning.

    Args:
        model(nn.Module): The model of the attacked client .
        data_dir(str): Directory of the dataset.
        n_classes(int): Number of classes in the dataset.
        device(str): Device on which to perform computations.
        dataset_name(str): The name of the datasets to use.
        batch_size(int): Batch size for a client update step.
        n_local_epochs(int): Number of local training epochs.
        learning_rate(float): Learning rate for the optimizer.
        seed(int): Seed for the fixing reproducibility.
        double_precision(bool): Whether to use double precision for the computations. Default is True.
    """

    def __init__(self, model, data_dir, device, dataset_name, batch_size, n_classes, n_local_epochs=1, learning_rate=1e-3, seed=42, double_precision=True):
        self.model = model
        self.device = device
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.seed = seed
        self.double_precision = double_precision
        self.data_dir = data_dir
        self.n_local_epochs = n_local_epochs
        self.inputs, self.labels = self._get_data()
        self.learning_rate = learning_rate

    
    def _get_data(self):
        """
        Load the first n_samples from the already prepared data sample.
        """

        data_path = os.path.join(self.data_dir, 'sample', f"{self.seed}")

        if not os.path.exists(data_path):
            raise FileNotFoundError("Data file does not exist. Please, indicate the sample to generate using flag '--generate_data'.")
        
        inputs = torch.load(os.path.join(data_path, 'inputs.pt'), weights_only=True)
        labels = torch.load(os.path.join(data_path, 'labels.pt'), weights_only=True)

        inputs = inputs[:self.batch_size]
        labels = labels[:self.batch_size]

        return inputs.to(self.device), labels.to(self.device)

    
    @abstractmethod
    def execute_attack(self):
        """
        Execute the attack on the provided dataset.
        """
        pass

    @abstractmethod
    def evaluate_attack(self):
        """
        Evaluate the attack on the provided dataset.
        """
        pass
    

    def simulate_communication_round(self, debug=False):
        """Simulate a communication round between the client and the server.
        Returns the updates that the server can observe to reconstruct the images.

        Args:
            debug(bool): Whether to print debug information. Default is False.
        Returns:
            observations(torch.Tensor): The observations of the server.
            sum_dL_db(torch.Tensor): The sum of the gradients of the loss with respect to the bias.
        """

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        if self.dataset_name == 'adult':
            criterion = nn.BCEWithLogitsLoss()
        elif self.dataset_name in ['cifar10', 'cifar100', 'tiny-imagenet', 'harus', 'imagenet']:
            criterion = nn.CrossEntropyLoss()
        self.model.train()
        for e in range(self.n_local_epochs):
            optimizer.zero_grad()
            for i in range(self.inputs.shape[0]):
                pred = self.model(self.inputs[i])

                pred = pred.squeeze()
                loss = criterion(pred, self.labels[i])

                if debug:
                    logging.info(f"b: {self.model.layers[0].bias.data}")
                    # logging.debug(f"db: {dL_db_all}")
                    # logging.debug(f"sum db: {torch.sum(dL_db_all, dim=0)}")


                loss.backward()
            optimizer.step()
            dL_db_input = self.model.layers[0].bias.grad.data.detach().clone()
            dL_dA_input = self.model.layers[0].weight.grad.data.detach().clone()

            if e == 0:
                sum_dL_dA = dL_dA_input
                sum_dL_dB = dL_db_input
            
            else:
                sum_dL_dA += dL_dA_input
                sum_dL_dB += dL_db_input
       
        obs_rec = sum_dL_dA / sum_dL_dB.view(-1, 1)
        return obs_rec.cpu(), sum_dL_dB.cpu()


    def remove_multiple_reconstruction(self, paired_inputs):
        """Remove duplicates of reconstruction of the same true image.
        Args:
            paired_inputs(list): List of tuples in the form (reconstructed_image, true_image, evaluation_metric)
        Returns:
            paired_inputs(list): List of paired inputs without duplicates"""
        
        cleaned_inputs = []
        for x in self.inputs:
            x = x.cpu()
            to_compare = []
            for i in paired_inputs:
                if torch.equal(i[1], x):
                    to_compare.append((i[0], i[2]))
            if len(to_compare) == 1:
                cleaned_inputs.append((to_compare[0][0], x, to_compare[0][1]))
            elif len(to_compare) > 1:
                to_compare.sort(key=lambda x: x[1])
                if self.dataset_type != 'tabular':
                    cleaned_inputs.append((to_compare[-1][0], x, to_compare[-1][1]))
                else:
                    cleaned_inputs.append((to_compare[0][0], x, to_compare[0][1]))
        return cleaned_inputs


    def get_tabular_evaluation(self, paired_inputs):
        """Return evaluation metrics dictionary for tabular datasets."""

        if len(paired_inputs) == 0:
            result_dict = {
            'max_norm_diff': 0,
            'avg_norm_diff': 0,
            'max_diff': 0,
            'n_perfect_reconstructed': 0,
            'norm_diff_list': []
        }
        else:
            norm_diff_list = [torch.norm(paired_inputs[i][0] - paired_inputs[i][1]).item() for i in range(len(paired_inputs))]
            diff_list = [paired_inputs[i][0] - paired_inputs[i][1] for i in range(len(paired_inputs))]
            avg_norm_diff = sum(norm_diff_list)/len(norm_diff_list)
            max_norm_diff = max(norm_diff_list)
            max_diff = max([torch.max(diff_list[i]).item() for i in range(len(diff_list))])
            n_perfect_reconstructed = sum([norm_diff_list[i] < 0.1 for i in range(len(norm_diff_list))])
            result_dict = {
                "norm_diff_list": np.array(norm_diff_list).astype(np.double).tolist(),
                "avg_norm_diff": avg_norm_diff,
                "max_norm_diff": max_norm_diff,
                "max_feat_diff": max_diff,
                "n_perfect_reconstructed": np.double(n_perfect_reconstructed)
            }
        return result_dict
    

    def get_image_evaluation(self, paired_inputs):
        """Return evaluation metrics dictionary for image datasets."""
        if len(paired_inputs) == 0:
            result_dict = {
            'avg_ssim': 0,
            'avg_psnr': 0,
            'n_perfect_reconstructed': 0,
            'ssim_list': [],
            'psnr_list': [],
            'norm_diff_list': []
        }
            
        else:
            ssim_list = [get_ssim(paired_inputs[i][0], paired_inputs[i][1], self.dataset_name) for i in range(len(paired_inputs))]
            avg_ssim = sum(ssim_list)/len(ssim_list)
            psnr_list = [get_psnr(paired_inputs[i][0], paired_inputs[i][1], data_range=2) for i in range(len(paired_inputs))]
            avg_psnr = sum(psnr_list)/len(psnr_list)
            norm_diff_list = [torch.norm(paired_inputs[i][0] - paired_inputs[i][1]).item() for i in range(len(paired_inputs))]
            n_perfect_reconstructed = sum([norm_diff_list[i] < 0.1 for i in range(len(norm_diff_list))])
            result_dict = {
                "ssim_list": np.array(ssim_list).astype(np.double).tolist(),
                "avg_ssim": avg_ssim,
                "psnr_list": np.array(psnr_list).astype(np.double).tolist(),
                "avg_psnr": avg_psnr,
                "norm_diff_list": np.array(norm_diff_list).astype(np.double).tolist(),
                "n_perfect_reconstructed": np.double(n_perfect_reconstructed)
            }
        return result_dict
    




class HyperplaneSampleReconstructionAttack(BaseSampleReconstructionAttack):
    """
    Class representing a sample reconstruction attack in federated learning.
    This attack aims to reconstruct the training samples of a target client in federated learning.

    Args:
        model(nn.Module): The model of the attacked client .
        device(str): Device on which to perform computations.
        dataset_name(torch.utils.data.Dataset): The dataset of the attacked client.
        n_classes(int): Number of classes in the dataset.
        seed(int): Seed for the fixing reproducibility.
        epsilon(float): Stopping condition for the search.
        atol(float): Absolute tolerance for checking if two observartions are equal.
        rtol(float): Relative tolerance for checking if two observartions are equal.
        batch_size(int): Batch size for a client update step.
        parallelize(bool): Whether to parallelize the model over multiple GPUs using DataParallel module. Defualt is False. 
    """

    def __init__(self, model, dataset_name, n_classes, device, seed, double_precision,
                 epsilon, atol, rtol, batch_size, data_dir, n_local_epochs=1, learning_rate=1e-3,parallelize=False):

        super(HyperplaneSampleReconstructionAttack, self).__init__(
            model=model, 
            data_dir=data_dir,
            device=device,
            dataset_name=dataset_name,
            seed=seed,
            n_classes=n_classes,
            batch_size=batch_size,
            double_precision=double_precision,
            n_local_epochs=n_local_epochs, 
            learning_rate=learning_rate
            )
        self.epsilon = epsilon
        self.atol = atol
        self.rtol = rtol
        self.dataset_type = "image" if dataset_name in ['cifar10', 'cifar100', 'imagenet'] else "tabular"

        self.model.to(self.device)
        self.parallelize = parallelize

        if self.double_precision:
            self.model = self.model.double()
        if self.parallelize and device != 'cuda':
            raise ValueError("Parallelization is only supported on GPUs. Please set the device to 'cuda'.")
        
        if self.parallelize and torch.cuda.device_count() <=1:
            raise ValueError("Parallelization is only supported on multiple GPUs.")
        
        elif self.parallelize:
            self.model = nn.DataParallel(self.model)

        self.round = 0
    
        self.intervals, self.spacing, self.inputs_idx = self._initialize_search_intervals()

        self.current_search_state = []

    
    def _initialize_search_intervals(self):
        """
        Initialize the search intervals for the attack.
        
        Returns:
            intervals(list): List of tuples representing the search intervals.
            spacing(float): The maximum intervals' distance.
        """
        n_hyperplanes = self.model.layers[0].bias.data.shape[0]
        b_tensor = - torch.matmul(self.model.layers[0].weight[0], torch.transpose(self.inputs, 0, 1)).cpu().detach()
        b_sorted, indices = torch.sort(torch.tensor(b_tensor.clone().detach()), dim=0)
        sorted_indices = torch.argsort(b_tensor, dim=0)
        if self.double_precision:
            b_1 = - torch.matmul(abs(self.model.layers[0].weight[0]).cpu(), torch.ones(self.inputs.shape[1]).double()).detach()
            b_2 = - torch.matmul(abs(self.model.layers[0].weight[0]).cpu(), (torch.ones(self.inputs.shape[1]).double() * -1)).detach()
        else:
            b_1 = - torch.matmul(abs(self.model.layers[0].weight[0]).cpu(), torch.ones(self.inputs.shape[1])).detach()
            b_2 = - torch.matmul(abs(self.model.layers[0].weight[0]).cpu(), (torch.ones(self.inputs.shape[1]) * -1)).detach()
        b_min = torch.min(b_1, b_2)
        b_max = torch.max(b_1, b_2)

        intervals = [(b_min, b_max)] 
        spacing = (b_max - b_min)/ n_hyperplanes

        return intervals, spacing, sorted_indices

    def _set_malicious_model_params(self):
        """
        Modify the model parameters to perform the attack.
        
        This method simulates the malicious server modifications of the model parameters.
        """

        n_hyperplanes = self.model.layers[0].bias.data.shape[0]

        if len(self.intervals) == 1:
            if round != 0:
                max_spacing = (self.intervals[0][1] - self.intervals[0][0]) / (n_hyperplanes + 2)
                self.model.layers[0].bias.data = torch.linspace(self.intervals[0][0] + max_spacing,
                                                                self.intervals[0][1] - max_spacing, n_hyperplanes)
            else: 
                max_spacing = (self.intervals[0][1] - self.intervals[0][0]) / n_hyperplanes
                self.model.layers[0].bias.data = torch.linspace(self.intervals[0][0], 
                                                                self.intervals[0][1], n_hyperplanes)
        else:

            self.intervals.sort(key=lambda x: x[1] - x[0], reverse=True)
            mod_hp = n_hyperplanes % len(self.intervals)
            n_hp = n_hyperplanes // len(self.intervals)
            curr_idx = 0
            bias_tensor = torch.zeros_like(self.model.layers[0].bias.data)
            max_spacing = 0
            interval_idx = 0

            while curr_idx < n_hyperplanes:
                curr_b = self.intervals[interval_idx][0].detach().clone()

                # case with the extra hyperplane
                remaining_hp = n_hyperplanes - curr_idx
                if interval_idx < mod_hp:
                    spacing = (self.intervals[interval_idx][1] - self.intervals[interval_idx][0]) / (n_hp + 2)
                    for j in range(min(n_hp + 1, remaining_hp)):
                        bias_tensor[curr_idx] = curr_b + spacing
                        curr_b += spacing
                        curr_idx += 1

                # case with no extra hyperplane
                else:
                    spacing = (self.intervals[interval_idx][1] - self.intervals[interval_idx][0]) / (n_hp + 1)
                    for j in range(min(n_hp , remaining_hp)):
                        bias_tensor[curr_idx] = curr_b + spacing
                        curr_b += spacing
                        curr_idx += 1

                if max_spacing < spacing:
                    max_spacing = spacing
                interval_idx += 1

            # order the bias tensor
            self.model.layers[0].bias.data, _ = torch.sort(bias_tensor)

            # reorder intervals
            self.intervals.sort(key=lambda x: x[0])

        self.model = self.model.to(self.device)

        if self.double_precision:
            self.model = self.model.double()

        if self.parallelize:
            self.model = nn.DataParallel(self.model)

    
    def _update_search_intervals(self):
        """
        Update the search intervals and search state, based on the current observations.
        """
        obs_history = sorted(self.current_search_state, key=lambda x: x[0])
        self.current_search_state = [] 
        new_intervals = []
        i = 0
        # skip multiple observations without any activation
        while torch.isnan(obs_history[i][1]).any():
            i = i + 1
        self.current_search_state.append(obs_history[i - 1])
        self.current_search_state.append(obs_history[i])
        i = i + 1
        # update observations
        while i < len(obs_history) -1:
            if (not torch.allclose(obs_history[i][1], obs_history[i-1][1], atol=self.atol, rtol=self.rtol)) or \
                (not torch.allclose(obs_history[i][1], obs_history[i+1][1], atol=self.atol, rtol=self.rtol)):
                self.current_search_state.append(obs_history[i])
            i = i + 1
        # handle the last observation
        if not torch.allclose(obs_history[-1][1], obs_history[-2][1], atol=self.atol, rtol=self.rtol):
            self.current_search_state.append(obs_history[-1])

        if len(self.current_search_state) > 2:
            i = 1
            int_start = self.current_search_state[0][0].detach().clone()
            while i < len(self.current_search_state) - 1:
                if not torch.allclose(self.current_search_state[i][1], 
                                      self.current_search_state[i-1][1], atol=self.atol, rtol=self.rtol):
                    int_end = self.current_search_state[i][0].detach().clone()
                    new_intervals.append((int_start, int_end))
                    int_start = None
                
                if not torch.allclose(self.current_search_state[i][1], 
                                      self.current_search_state[i+1][1], atol=self.atol, rtol=self.rtol):
                    int_start = self.current_search_state[i][0].detach().clone()
                i = i + 1
            if int_start is not None:
                int_end = self.current_search_state[-1][0]
                new_intervals.append((int_start, int_end))

        else:
            new_intervals.append((self.current_search_state[0][0].detach().clone(), self.current_search_state[1][0].detach().clone()))

        if len(new_intervals) > 1:
            self.intervals = [i for i in new_intervals if i[1] - i[0] > self.epsilon]
        else:
            self.intervals = new_intervals

    def _extract_observations(self):
        """
        Extract the observations from the search state.
        """

        search_state = copy.deepcopy(self.current_search_state)
        if torch.isnan(search_state[0][1]).any():
            search_state.pop(0)
            
        final_observations_list = [search_state[0][1]]
        coeffs_list = [search_state[0][2]]

        for i in range(1, len(search_state)):
            if not torch.allclose(search_state[i][1], 
                                  search_state[i-1][1], atol=self.atol, rtol=self.rtol):
                final_observations_list.append(search_state[i][1])
                coeffs_list.append(search_state[i][2])
        
        return final_observations_list, coeffs_list
        
    def _reconstruct_samples(self, observations_list, coeffs_list):
        """
        Reconstruct the samples based on the observations and the coefficients.
        
        Args:
            observations_list(list): List of observations.
            coeffs_list(list): List of coefficients.
        Returns:
            rec_inputs(torch.Tensor): The reconstructed samples.
        """
        rec_inputs = [observations_list[0]]
        dL_db_list = [coeffs_list[0]]

        for k in range(1, len(coeffs_list)):
            obs = observations_list[k]
            dL_db_k = coeffs_list[k] - coeffs_list[k-1]
            dL_db_list.append(dL_db_k)
            alphas = [i/coeffs_list[k] for i in dL_db_list]
            # print(f"Alphas for round {k}: {alphas}")
            recon_input = (obs - sum([alphas[i] * rec_inputs[i] for i in range(len(rec_inputs))]))/ alphas[-1]
            rec_inputs.append(recon_input)

        return rec_inputs
        

    def execute_attack(self, eval_round=None, debug=False):
        """
        Execute the attack on the provided client's samples.
        The attack performs the following steps:
        1. Simulate a communication round.
        2. Server computes the new model parameters and search interval.
        3. When the stopping condition is met, the communication stops.
        4. Server reconstructs the samples, according to the observations.
        5. Server returns the reconstructed samples.

        Args:
            eval_round(int): Round to evaluate. Default is None.
            debug(bool): Whether to print debug information. Default is False.
        """

        if debug:
            b_tensor = - torch.matmul(self.model.layers[0].weight[0], torch.transpose(self.inputs, 0, 1)).cpu().detach()
            b_sorted, indices = torch.sort(torch.tensor(b_tensor.clone()), dim=0)
            logging.debug(f"Min spacing between inputs: {torch.min(b_sorted[1:] - b_sorted[:-1])}")

        while self.spacing >= self.epsilon or self.round == 0:
            logging.info("-------------------")
            logging.info(f"Simulating round {self.round}")
            logging.info(f"Current spacing: {self.spacing}")

            self._set_malicious_model_params()

            # copy the model before the communication round
            server_model = copy.deepcopy(self.model.state_dict())

            observations, sum_dL_db = self.simulate_communication_round(debug=debug) 

            # server reloads the old model 
            self.model.load_state_dict(server_model)
            
            self.current_search_state.extend((self.model.layers[0].bias.data[i].detach().clone().cpu(), 
                                              observations[i], sum_dL_db[i].item() ) for i in range(observations.shape[0]))
            
            self._update_search_intervals()

            self.round += 1
            if len(self.intervals) == 0:
                self.spacing = 0
            else:
                self.spacing = self.intervals[0][1] - self.intervals[0][0]

            if self.round == eval_round:
                observations_list, coeffs_list = self._extract_observations() 

                rec_inputs = self._reconstruct_samples(observations_list, coeffs_list)

                return rec_inputs
            
    
    def evaluate_attack(self, rec_input):
        """
        Evaluate the attack.
        """
        """
        Args:
            rec_input(list): List of reonstructed inputs.
            
        """
        
        if len(rec_input) == 0:
            logging.info("No image was reconstructed.")
            paired_inputs = []
            
        elif len(rec_input) == self.inputs.shape[0]:
            if self.dataset_type == 'tabular':
                paired_inputs = [(rec_input[i], self.inputs[self.inputs_idx[i]].cpu(), torch.norm(rec_input[i] - self.inputs[self.inputs_idx[i]].cpu()) ) \
                                  for i in range(len(rec_input))]
            else:
                paired_inputs = [(rec_input[i], self.inputs[self.inputs_idx[i]].cpu(), get_ssim(rec_input[i], self.inputs[self.inputs_idx[i]].cpu(), self.dataset_name) ) \
                                  for i in range(len(rec_input))]
        else:
            inputs_list = [self.inputs[i].cpu() for i in range(self.inputs.shape[0])]
            paired_inputs = couple_inputs(rec_input, inputs_list, dataset_name=self.dataset_name)
            paired_inputs = self.remove_multiple_reconstruction(paired_inputs)
            
        if self.dataset_type == 'tabular':
            result_dict = self.get_tabular_evaluation(paired_inputs)
        else:
            result_dict = self.get_image_evaluation(paired_inputs)

        logging.info(f"Number of observations: {len(paired_inputs)}")
        logging.info(f"Number of perfect reconstructions: {result_dict['n_perfect_reconstructed']}")

        return result_dict
    
class CuriousAbandonHonestyAttack(BaseSampleReconstructionAttack):
    """
    Class implementing the Currious Abandon Honesty attack.
    """
    
    def __init__(self, model, device, dataset_name, n_classes, seed, double_precision, batch_size, data_dir, atol, rtol, learning_rate=1e-3, n_local_epochs=1, parallelize=False):
            super(CuriousAbandonHonestyAttack, self).__init__(
                model=model, 
                data_dir=data_dir,
                device=device,
                dataset_name=dataset_name,
                n_classes=n_classes,
                seed=seed,
                double_precision=double_precision,
                batch_size=batch_size,
                n_local_epochs=n_local_epochs,
                learning_rate=learning_rate
                )
            
            self.atol = atol
            self.rtol = rtol
            self.parallelize = parallelize
            self.current_search_state = []
            self.dataset_type = "image" if dataset_name in ['cifar10', 'cifar100', 'imagenet'] else "tabular"
            self.round = 0

            self.model.to(self.device)
            if self.double_precision:
                self.model = self.model.double()
            if self.parallelize and device != 'cuda':
                raise ValueError("Parallelization is only supported on GPUs. Please set the device to 'cuda'.")
        
            if self.parallelize and torch.cuda.device_count() <=1:
                raise ValueError("Parallelization is only supported on multiple GPUs.")
        
            elif self.parallelize:
                self.model = nn.DataParallel(self.model)
    

    def execute_attack(self, sigma, mu, scale_factor, eval_round=1):
        """
        Execute the Curious Abandon Honesty Attack.

        1. Set the malicious model parameters
        2. Simulate a communication round
        3. Retrieve the observations
        """

        while self.round  < eval_round:
            logging.info("-------------------")
            logging.info(f"Simulating round {self.round}")
            self._set_malicious_model_params(sigma=sigma, mu=mu, scale_factor=scale_factor)

            observations, _  = self.simulate_communication_round()

            self.current_search_state.extend(observations)

            self._clean_search_state()
            logging.info(f"N. isolated images: {len(self.current_search_state)}")

            self.round += 1

        observations_list = self._extract_observations()
        return observations_list  
    

    def _clean_search_state(self):
        """
        Remove observations that are not reconstructed inputs. 
        We use the L2 norm to determine if two observations are the same, to keep the same evaluation metric as

        """

        new_search_state = []
        for obs in self.current_search_state:
            for i in range(self.inputs.shape[0]):
                if torch.norm(obs - self.inputs[i].cpu()) < 0.1:
                    new_search_state.append(obs)
                    break

        # remove duplicates
        cleaned_list = list(set(new_search_state))

        self.current_search_state = cleaned_list
        
            

    def _set_malicious_model_params(self, sigma, mu, scale_factor):
        """" Set the malicious model parameters"""
        N, K = self.inputs.shape[1], self.model.layers[0].weight.shape[0]

        indices = torch.zeros((K, N), dtype=torch.long)
        for row in range(K):
            indices[row] = torch.randperm(N)
        negative_weight_indices = indices[:, : int(N / 2)]
        positive_weight_indices = indices[:, int(N / 2) :]

        if N % 2 != 0:
            sampled_weights = torch.abs(torch.randn(K, int(math.ceil(N / 2))) * sigma) * -1

            negative_samples = sampled_weights[:, :-1]
        else:

            sampled_weights = torch.abs(torch.randn(K, int(N / 2)) * sigma) * -1
            negative_samples = sampled_weights
        positive_samples = -scale_factor * sampled_weights

        final_weights = torch.empty(K, N)
        final_weights.scatter_(1, negative_weight_indices, negative_samples)
        final_weights.scatter_(1, positive_weight_indices, positive_samples)
        if self.double_precision:
            final_weights = final_weights.double()
        self.model.layers[0].weight = nn.Parameter(final_weights)

        bias_tensor = torch.ones_like(self.model.layers[0].bias) * mu
        if self.double_precision:
            bias_tensor = bias_tensor.double()
        self.model.layers[0].bias = nn.Parameter(bias_tensor)

        self.model = self.model.to(self.device)


    def _is_similar(self, t1, t2):
                return torch.allclose(t1, t2, atol=self.atol, rtol=self.rtol)
    
    def _extract_observations(self):
        """
        Extract the observations from the search state.
        """
        cleaned_list = [self.current_search_state[i] for i in range(len(self.current_search_state)) if not torch.isnan(self.current_search_state[i]).any()]

        unique_obs = []
        for obs in cleaned_list:
            if not any(self._is_similar(obs, unique_tensor) for unique_tensor in unique_obs):
                unique_obs.append(obs)
        return unique_obs
    

    def evaluate_attack(self, rec_input):
        """
        Evaluate the attack.
        """
        """
        Args:
            rec_input(list): List of reonstructed inputs.
            
        """
        
        if len(rec_input) == 0:
            logging.info("No image was reconstructed.")
            paired_inputs = []
    
        else:
            inputs_list = [self.inputs[i].cpu() for i in range(self.inputs.shape[0])]
            paired_inputs = couple_inputs(rec_input, inputs_list, dataset_name=self.dataset_name)
            paired_inputs = self.remove_multiple_reconstruction(paired_inputs)
            
        if self.dataset_type == 'tabular':
            result_dict = self.get_tabular_evaluation(paired_inputs)
        else:
            result_dict = self.get_image_evaluation(paired_inputs)

        logging.info(f"Number of observations: {len(paired_inputs)}")
        logging.info(f"Number of perfect reconstructions: {result_dict['n_perfect_reconstructed']}")

        return result_dict
