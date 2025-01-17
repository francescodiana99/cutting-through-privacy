import json
import logging
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import itertools

import matplotlib.pyplot as plt

import argparse
import os

from utils_search import *
from utils_misc import *
from utils_test import * 

from attacks.sra import HyperplaneSampleReconstructionAttack, CuriousAbandonHonestyAttack
from models.resnets import ResNet, resnet_depths_to_config


INPUT_DIM = {
    "imagenet": 3*224*224,
    "cifar10": 3*32*32,
    "cifar100": 3*32*32,
    "adult": 97,
    "harus": 561
             }
N_CLASSES = {
    "imagenet": 1000,
    "cifar10": 10,
    "cifar100": 100,
    "adult": 1,
    "harus": 6
                }

def parse_args():
    """
    Parse the arguments for the script.
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir",
                        type=str,
                        default='./data/',
                        help="Path to the data directory."
                        )
    
    parser.add_argument("--n_samples",
                        type=int,
                        help="Number of samples to use for the search."
                        )
    
    parser.add_argument("--n_rounds",
                        type=int,
                        help="Number of rounds for the search."
                        )
    
    parser.add_argument("--eval_freq",
                        type=int,
                        help="Evaluation frequency for the attack.")
    
    parser.add_argument("--device",
                        type=str,
                        default='cpu',
                        help="Device to use for the computation."
                        )
                        
    
    parser.add_argument("--display",
                        action='store_true',
                        help="Display the images.",
                        default=False
                        )
    
    parser.add_argument("--dataset",
                        type=str,
                        default='cifar10',
                        help="Dataset to use for the experiment. \
                        Possible values are 'cifar10', 'tiny-imagenet' and 'adult'.")
    
    
    parser.add_argument("--hidden_layers",
                        nargs='+' ,
                        type=int,
                        help="Hidden layers structure of the neural network."
                        )
    
    parser.add_argument("--classification_bias", 
                        type=float,
                        default=1e12,
                        help="Classification bias value"
                        ),
    
    parser.add_argument("--input_weights_scale",
                        type=float,
                        default=1,
                        help="Scale factor for the input weights initialization."
                        )
    
    parser.add_argument("--hidden_weights_scale",
                        type=float,
                        default=1e-3,
                        help="Scale factor for the hidden weights initialization."
                        )
    
    parser.add_argument("--hidden_bias_scale",
                        type=float,
                        default=1e-3,
                        help="Scale factor for the hidden bias initialization."
                        )
    
    parser.add_argument("--classification_weight_scale",
                        type=float,
                        default=1,
                        help="Scale factor for the classification weights initialization."
                        )
                        
    
    parser.add_argument("--double_precision",
                        action='store_true',
                        help="Use double precision for the computation.",
                        default=False)
    
    parser.add_argument("--epsilon",
                        type=float,
                        default=1e-14,
                        help="Epsilon value for the attack."
                        )
    
    parser.add_argument("--atol",
                        type=float,
                        default=1e-5,
                        help="Absolute tolerance value for the attack."
                        )
    
    parser.add_argument("--rtol",
                        type=float,
                        default=1e-6,
                        help="Relative tolerance value for the attack."
                        )
    
    parser.add_argument("--debug",
                        action='store_true',
                        help="Flag for debugging option.",
                        default=False)
    
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Seed value for the random number generator."
                        )
    
    parser.add_argument("--results_path",
                        type=str,
                        default='./results',
                        help="Path to save the results."
                        )
    
    parser.add_argument("--save_reconstruction",
                        action='store_true',
                        help="Flag to save the reconstructed images.",
                        default=False
                        )
    

    parser.add_argument("--attack_name",
                        type=str,
                        help="Name of the attack to use."
                        )
    parser.add_argument("--mu",
                        type=float,
                        help="Mean of the normal distribution for CaH attack."
                        )
    parser.add_argument("--sigma",
                        type=float,
                        help="Standard deviation of the normal distribution for CaH attack."
                        )
    
    parser.add_argument("--scale_factor",
                        type=float,
                        help="Scale factor for the CaH attack."
                        )   
    
    
    
    return parser.parse_args()


def main():

    args = parse_args()

    set_seeds(args.seed)

    configure_logging()

    dataset_type = "image" if args.dataset in ['cifar10', 'cifar100', 'tiny-imagenet', 'imagenet'] else 'tabular'

    n_classes = N_CLASSES[args.dataset]
    input_dim = INPUT_DIM[args.dataset]

    if args.attack_name == 'hsra':
        model = FCNet(
            input_dimension=input_dim, 
            output_dimension=n_classes, 
            classification_weight_scale=args.classification_weight_scale,
            hidden_layers=args.hidden_layers,
            classification_bias=args.classification_bias,
            input_weights_scale=args.input_weights_scale, 
            hidden_weights_scale=args.hidden_weights_scale, 
            hidden_bias_scale=args.hidden_bias_scale,
            honest=False
            )

        sra_attack = HyperplaneSampleReconstructionAttack(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            model=model,
            device=args.device,
            seed=args.seed,
            epsilon=args.epsilon,
            atol=args.atol,
            rtol=args.rtol,
            double_precision=args.double_precision,
            batch_size=args.n_samples,
            parallelize=False, 
        )

        rec_input = sra_attack.execute_attack(debug=args.debug, n_rounds=args.n_rounds)

    
    elif args.attack_name == 'cah':
        model = FCNet(
            input_dimension=input_dim, 
            output_dimension=n_classes,
            hidden_layers=args.hidden_layers,
            honest=True)
        
        sra_attack = CuriousAbandonHonestyAttack(
            model=model,
            device=args.device,
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            batch_size=args.n_samples,
            seed=args.seed,
            double_precision=args.double_precision,
            atol=args.atol,
            rtol=args.rtol,
            parallelize=False
        )
        rec_input = sra_attack.execute_attack(n_rounds=args.n_rounds, mu=args.mu, sigma=args.sigma, scale_factor=args.scale_factor)

    else:
        raise ValueError(f"Attack name {args.attack_name} is not supported.")


    inputs_list = [sra_attack.inputs[i].cpu() for i in range(sra_attack.inputs.shape[0])]

    logging.info("Reconstruction completed. Pairing samples with true images...")
    # it means we have all the images
    if len(rec_input) == sra_attack.inputs.shape[0] and args.attack_name == 'hsra':
        paired_inputs = [(rec_input[i], sra_attack.inputs[sra_attack.inputs_idx[i]].cpu()) for i in range(len(rec_input))]
    else:
        paired_inputs = couple_inputs(rec_input, inputs_list, dataset_name=args.dataset)
    if args.display:
        restore_images([paired_inputs[0][0], paired_inputs[0][1].cpu()], device=args.device, display=True)
    
    if dataset_type != 'tabular':
        ssim_list, avg_ssim, psnr_list, avg_psnr, n_perfect_reconstructed = sra_attack.evaluate_attack(paired_inputs,
                                                                            dataset_name=args.dataset)
        logging.info("-----------Metrics-----------")
        logging.info(f"Number of samples: {len(paired_inputs)}")
        logging.info(f"Number of perfect reconstructions: {n_perfect_reconstructed}")
        logging.info(f"Average SSIM: {avg_ssim}")
        logging.info(f"Average PSNR: {avg_psnr}")
        
    else:
        norm_diff_list, avg_norm_diff, max_norm_diff, max_diff, n_perfect_reconstructed = sra_attack.evaluate_attack(paired_inputs)
        logging.info("-----------Metrics-----------")
        logging.info(f"Number of samples: {len(paired_inputs)}")
        logging.info(f"Number of perfect reconstructions: {n_perfect_reconstructed}")
        logging.info(f"Max diff: {max_diff}")
        logging.info(f"Max norm diff: {max_norm_diff}")
        logging.info(f"Average  norm diff: {avg_norm_diff}")

        
    os.makedirs(args.results_path, exist_ok=True)
    if args.save_reconstruction:
        rec_images_path = os.path.join(args.results_path, 'reconstructed.pt')
        torch.save(torch.stack(rec_input), rec_images_path)

    if dataset_type != 'tabular':
        result_dict = {
            'ssim_list': np.array(ssim_list).astype(np.double).tolist(),
            'avg_ssim': avg_ssim,
            'psnr_list': np.array(psnr_list).astype(np.double).tolist(),
            'avg_psnr': avg_psnr,
            'n_perfect_reconstructed': n_perfect_reconstructed
        }
    else:
        result_dict = {
            'max_norm_diff': max_norm_diff,
            'avg_norm_diff': avg_norm_diff,
            'max_diff': max_diff,
            'n_perfect_reconstructed': n_perfect_reconstructed,
            'l2_norm_diff_list': np.array(norm_diff_list).astype(np.double).tolist()
        }
    results_path = os.path.join(args.results_path, 'results.json')
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        results[f"{args.n_rounds}"] = result_dict
    else:
        results = {f"{args.n_rounds}": result_dict}
    with open(results_path, 'w') as f:
        json.dump(results, f)
    logging.info(f"Results saved in {results_path}.")

    

if __name__ == '__main__':
    main()
