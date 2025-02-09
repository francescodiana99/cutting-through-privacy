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
from models.conv_net import ConvNet


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
    
    parser.add_argument("--eval_rounds",
                        nargs='+',
                        type=int,
                        help="Evaluation rounds for the attack.")
    
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
                        
    
    parser.add_argument("--generate_samples",
                        type=int,
                        help="Number of samples to generate for the attack.")
    
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-3,
                        help="Learning rate for the attack."
                        )
    
    parser.add_argument("--n_local_epochs",
                        type=int,
                        default=100,
                        help="Number of local training epochs ."
    )

    parser.add_argument("--use_batch_computation",
                        action='store_true',
                        help="Flag to use batch computation.",
                        default=False)
    
    parser.add_argument("--model_type",
                        type=str,
                        default='fc',
                        help="Type of the model to use for the attack."
                        )
    
    
    return parser.parse_args()


def main():

    args = parse_args()

    set_seeds(args.seed)

    configure_logging()

    n_classes = N_CLASSES[args.dataset]
    input_dim = INPUT_DIM[args.dataset]

    eval_rounds = args.eval_rounds
    eval_rounds.sort()

    if args.generate_samples is not None:
        inputs, labels, _ = prepare_data(dataset=args.dataset, 
                                                 data_dir=args.data_dir, 
                                                 n_samples=args.generate_samples,
                                                 double_precision=args.double_precision
                                                 )
    # save the generated samples
        inputs_path = os.path.join(args.data_dir, 'sample', f"{args.seed}", 'inputs.pt')
        labels_path = os.path.join(args.data_dir, 'sample', f"{args.seed}", 'labels.pt')

        os.makedirs(os.path.join(args.data_dir, 'sample', f"{args.seed}"), exist_ok=True)
        torch.save(inputs, inputs_path)
        torch.save(labels, labels_path)
        logging.info(f"Generated samples saved in {inputs_path} and {labels_path}.")
    

    if args.attack_name == 'hsra':

        if args.model_type == 'fc':
            model = FCNet(
                input_dimension=input_dim, 
                output_dimension=n_classes, 
                classification_weight_scale=args.classification_weight_scale,
                hidden_layers=args.hidden_layers,
                classification_bias=args.classification_bias,
                input_weights_scale=args.input_weights_scale, 
                hidden_weights_scale=args.hidden_weights_scale, 
                hidden_bias_scale=args.hidden_bias_scale,
                honest=False,
                )
        elif args.model_type == 'cnn':
            model = ConvNet(
                num_classes=n_classes,
                in_channels=3,
                n_attack_neurons=1000,
                classification_weight_scale=args.classification_weight_scale,
                classification_bias_scale=args.classification_bias,
                attack_weight_scale=args.hidden_weights_scale,
                input_dim=input_dim
            )
        else:
            raise ValueError(f"Model type {args.model_type} is not supported.")
        
        sra_attack = HyperplaneSampleReconstructionAttack(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            model=model,
            model_type=args.model_type,
            device=args.device,
            seed=args.seed,
            epsilon=args.epsilon,
            atol=args.atol,
            rtol=args.rtol,
            double_precision=args.double_precision,
            batch_size=args.n_samples,
            n_classes=n_classes,
            parallelize=False, 
            learning_rate=args.learning_rate,
            n_local_epochs=args.n_local_epochs,
            batch_computation=args.use_batch_computation
        )
        for r in  eval_rounds:
            rec_input = sra_attack.execute_attack(debug=args.debug, eval_round=r)
            logging.info(f"Starting evaluation for round {r}...")
            result_dict = sra_attack.evaluate_attack(rec_input)

            if args.save_reconstruction:
                save_results(args.results_path, result_dict, r, rec_input)
            else:
                save_results(args.results_path, result_dict, r)
            
            logging.info(f"Results for round {r} saved in {args.results_path}.")

    
    elif args.attack_name == 'cah':
        if args.model_type == 'fc':
            model = FCNet(
                input_dimension=input_dim, 
                output_dimension=n_classes,
                hidden_layers=args.hidden_layers,
                honest=True)
        elif args.model_type == 'cnn':
            model = ConvNet(
                num_classes=n_classes,
                in_channels=3,
                n_attack_neurons=1000,
                classification_weight_scale=1,
                classification_bias_scale=1,
                attack_weight_scale=1,
                input_dim=input_dim
            )
        else:
            raise ValueError(f"Model type {args.model_type} is not supported.")
        
        sra_attack = CuriousAbandonHonestyAttack(
            model=model,
            model_type=args.model_type,
            device=args.device,
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            batch_size=args.n_samples,
            seed=args.seed,
            double_precision=args.double_precision,
            atol=args.atol,
            rtol=args.rtol,
            parallelize=False, 
            n_classes=n_classes,
            learning_rate=args.learning_rate,
            n_local_epochs=args.n_local_epochs,
            batch_computation=args.use_batch_computation
        )
        for r in eval_rounds:
            rec_input = sra_attack.execute_attack(eval_round=r, mu=args.mu, sigma=args.sigma, scale_factor=args.scale_factor)
            logging.info(f"Starting evaluation for round {r}...")
            result_dict = sra_attack.evaluate_attack(rec_input)

            if args.save_reconstruction:
                save_results(args.results_path, result_dict, r, rec_input)
            else:
                save_results(args.results_path, result_dict, r)
            
            logging.info(f"Results for round {r} saved in {args.results_path}.")

    else:
        raise ValueError(f"Attack name {args.attack_name} is not supported.")
    
if __name__ == '__main__':
    main()
