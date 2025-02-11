# Cutting Through Privacy: A Hyperplane-Based Data Reconstruction Attack in Federated Learning



## Getting started

This repo contains the code to run the experiments for the paper **Cutting Through Privacy: A Hyperplane-Based Data Reconstruction Attack in Federated Learning**. The scope of the attack is to reconstruct data sample from gradients in Federated Learning.

## Getting Started
Install the required packages using
```
pip install requirements.txt
```

## Usage
To reproduce our experiments on ImageNet first you have to download the [ILSVRC 2012 dataset](https://image-net.org/download-images.php) and place it in your `data_dir`. Then you can run the following command to launch our attack:
```
python run_attack.py --device cuda \ 
--n_samples 1024 \ 
--eval_rounds "1 2 5 10" \
--seed 3 \
--hidden_layers 1000 \
--dataset imagenet  \
--classification_bias 1e12 \ 
--input_weights_scale 1e-10 \
--hidden_weights_scale 1e-3 \
--n_local_epochs 1 \
--learning_rate 1e-10 \
--attack_name hsra   \
--hidden_bias_scale 1e-3 \
--classification_weight_scale 1e-2 \
--double_precision \
--epsilon 0 \
--atol 1e-5 \
--rtol 1e-7 \ 
--generate_samples 4000 \
--data_dir ~/data/imagenet
--results_path ./results/imagenet/

```
Add the flag `--save_reconstruction` to save the reconstructed images.

To run the CAH attack, use:
```
python run_attack.py --device cuda \ 
--n_samples 1024 \ 
--eval_rounds "1 2 5 10" \
--seed 3 \
--hidden_layers 1000 \
--dataset imagenet  \
--n_local_epochs 1 \
--learning_rate 1e-10 \
--attack_name cah   \
--double_precision \
--epsilon 0 \
--atol 1e-5 \
--rtol 1e-7 \ 
--mu 0 \
--sigma 0.5 \
--scale_factor 0.99 \
--generate_samples 4000 \
--data_dir ~/data/imagenet
--results_path ./results/imagenet/

```

To reproduce our esperiments on HARUS dataset, run:

```
python run_attack.py --device cuda 
--n_samples 4096
--eval_rounds "1 10 20 30 40 50" \
--seed 3 \
--hidden_layers 1000 \
--dataset harus \
--classification_bias 1e13 \
--input_weights_scale 1e-10 \
--hidden_weights_scale 1e-3 \
--n_local_epochs $n_local_epochs \
--learning_rate $learning_rate \
--hidden_bias_scale 1e-3 \
--classification_weight_scale 1e-4 \
--double_precision \
--epsilon 0 \
--atol 1e-5 \
--rtol 1e-7 \
--data_dir ~/data/harus  \
--results_path ./results/harus/ \
--attack_name hsra \
--generate_samples 8000

```
To run CAH attack, use:

```
--n_samples 4096
--eval_rounds "1 10 20 30 40 50" \
--seed 3 \
--hidden_layers 1000 \
--dataset harus \
--n_local_epochs $n_local_epochs \
--learning_rate $learning_rate \
--mu 0 \
--sigma 1 \
--scale_factor 0.97 \
--double_precision \
--epsilon 0 \
--atol 1e-5 \
--rtol 1e-7 \
--data_dir ~/data/harus  \
--results_path ./results/harus/ \
--attack_name cah \
--generate_samples 8000

```
The first time you run the attack, you should add the flag 
To modify the number of hidden layers, assign to `--hidden_layers`a list of number of neurons in each hidden layer, separated by space. For example using `--hidden_layers 1000 100 100` will train a neural network with 3 hidden layers, of 1000, 100 and 100 neurons respectively.



