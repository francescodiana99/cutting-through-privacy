# Cutting Through Privacy: A Hyperplane-Based Data Reconstruction Attack in Federated Learning
This repo contains the code to reproduce the experiments for the paper [Cutting Through Privacy: A Hyperplane-Based Data Reconstruction Attack in Federated Learning](https://dl.acm.org/doi/10.5555/3762387.3762431).  

Federated Learning (FL) enables collaborative training of machine learning models across distributed clients without sharing raw data, ostensibly preserving data privacy. Nevertheless, recent studies have revealed critical vulnerabilities in FL, showing that a malicious central server can manipulate model updates to reconstruct clients' private training data. Existing data reconstruction attacks have important limitations: they often rely on assumptions about the clients' data distribution or their efficiency significantly degrades when batch sizes exceed just a few tens of samples.  
In this work, we introduce a novel data reconstruction attack that overcomes these limitations. Our method leverages a new geometric perspective on fully connected layers to craft malicious model parameters, enabling the perfect recovery of arbitrarily large data batches in classification tasks without any prior knowledge of clients' data. Through extensive experiments on both image and tabular datasets, we demonstrate that our attack outperforms existing methods and achieves perfect reconstruction of data batches two orders of magnitude larger than the state of the art.

## Getting Started
Install the required packages using
```
pip install requirements.txt
```

## Usage
To reproduce our experiments on ImageNet, first you have to download the [ILSVRC 2012 dataset](https://image-net.org/download-images.php) and place it in your `data_dir`. Then you can run the following command to launch our attack:
```
python run_attack.py --device cuda \ 
--n_samples 1024 \ 
--eval_rounds 1 2 5 10 \
--seed 3 \
--hidden_layers 1000 \
--dataset imagenet  \
--classification_bias 1e25 \ 
--input_weights_scale 1e-2\
--hidden_weights_scale 1e-3 \
--n_local_epochs 1 \
--attack_name hsra   \
--hidden_bias_scale 1e-3 \
--classification_weight_scale 1e-1 \
--double_precision \
--epsilon 0 \
--atol 1e-5 \
--rtol 1e-7 \ 
--generate_samples 4000 \
--data_dir ~/data/imagenet \
--results_path ./results/imagenet/ \
--use_batch_computation

```
Add the flag `--save_reconstruction` to save the reconstructed images.

To run the CAH attack, use:
```
python run_attack.py --device cuda \ 
--n_samples 1024 \ 
--eval_rounds 1 2 5 10 \
--seed 3 \
--hidden_layers 1000 \
--dataset imagenet  \
--n_local_epochs 1 \
--attack_name cah   \
--double_precision \
--epsilon 0 \
--atol 1e-5 \
--rtol 1e-7 \ 
--mu 0 \
--sigma 0.5 \
--scale_factor 0.99 \
--generate_samples 4000 \
--data_dir ~/data/imagenet \
--results_path ./results/imagenet/ \
--use_batch_computation

```

To reproduce our experiments on HARUS dataset, run:

```
python run_attack.py \
--device cuda \
--n_samples 4096 \
--eval_rounds 1 10 20 30 40 50 \
--seed 3 \
--hidden_layers 1000 \
--dataset harus \
--classification_bias 1e25 \
--input_weights_scale 1e-1 \
--n_local_epochs 1 \
--classification_weight_scale 1e-1 \
--double_precision \
--epsilon 0 \
--atol 1e-5 \
--rtol 1e-7 \
--data_dir ~/data/harus  \
--results_path ./results/harus/ \
--attack_name hsra \
--generate_samples 8000 \
--use_batch_computation 


```
To run CAH attack, use:

```
python run_attack.py \
--device cuda \
--n_samples 4096 \
--eval_rounds 1 10 20 30 40 50 \
--seed 3 \
--hidden_layers 1000 \
--dataset harus \
--n_local_epochs 1 \
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
--generate_samples 8000 \
--use_batch_computation

```
The first time you run the attack, you should add the flag `generate_samples`
To modify the number of hidden layers, assign to `--hidden_layers`a list of number of neurons in each hidden layer, separated by space. For example using `--hidden_layers 1000 100 100` will train a neural network with 3 hidden layers, of 1000, 100 and 100 neurons respectively.

To run our attack on CIFAR-10, use:

```
python run_attack.py --device cuda \
--n_samples 2048 \
--eval_rounds 1 10 20 30 40 50 \
--seed 3 \
--hidden_layers 1000 \
--dataset cifar10 \
--classification_bias 1e25 \
--input_weights_scale 1e-1 \
--n_local_epochs 1 \
--classification_weight_scale 1e-1 \
--double_precision \
--epsilon 0 \
--atol 1e-5 \
--rtol 1e-7 \
--data_dir ~/data/cifar10  \
--results_path ./results/cifar10/ \
--attack_name hsra \
--generate_samples 4000 \
--use_batch_computation

```

To run CAH attack, use:

```
python run_attack.py \
--device cuda \
--n_samples 2048 \
--eval_rounds 1 10 20 30 40 50 \
--seed 3 \
--hidden_layers 1000 \
--dataset cifar10 \
--n_local_epochs 1 \
--mu 0 \
--sigma 1 \
--scale_factor 0.95 \
--double_precision \
--epsilon 0 \
--atol 1e-5 \
--rtol 1e-7 \
--data_dir ~/data/cifar10  \
--results_path ./results/cifar10/ \
--attack_name cah \
--generate_samples 8000 \
--use_batch_computation
```
Note, to run the attacks in single precision, remove the flag `--double_precision`.

Finally, to test the attacks on CNN, run:

```
python run_attack.py --device cuda \
--n_samples 2048 \
--eval_rounds 1 10 20 30 40 50 \
--seed 3 \
--hidden_layers 1000 \
--dataset cifar10 \
--classification_bias 1e25 \
--input_weights_scale 1e-1 \
--n_local_epochs 1 \
--classification_weight_scale 1e-1 \
--epsilon 0 \
--atol 1e-5 \
--rtol 1e-7 \
--data_dir ~/data/cifar10  \
--results_path ./results/cifar10/ \
--attack_name hsra \
--generate_samples 4000 \
--use_batch_computation \
--model_type cnn
```


```
python run_attack.py \
--device cuda \
--n_samples 32 \
--eval_rounds 1 10 20 30 40 50 \
--seed 42 \
--hidden_layers 1000 \
--dataset cifar10 \
--n_local_epochs 1 \
--mu 0 \
--sigma 1 \
--scale_factor 0.95 \
--epsilon 0 \
--atol 1e-5 \
--rtol 1e-7 \
--data_dir ~/data/cifar10  \
--results_path ./results/cifar10/ \
--attack_name cah \
--generate_samples 1000 \
--use_batch_computation \
--model_type cnn
```

Code to reproduce experiments with multiple local steps and noise will be added soon.
For any question, please, send an email to [francesco.diana@inria.fr](mailto:francesco.diana@inria.fr)





