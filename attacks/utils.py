"""This file contains utility functions and classes for attacks."""
import torch
import torch.nn as nn
from models.resnets import ResNet, resnet_depths_to_config
import torchvision

class MaliciousBlock(torch.nn.Module):
    """
    A linear block that can be placed in a neural network to perform a reconstruction attack.
    """
    def __init__(self, data_shape, n_neurons, weights_scale=1):
        """
        Args:
            data_shape(tuple): Shape of the input data.
            n_neurons(int): Number of neurons in the hidden layer.
            weights_scale(float): Scale factor for the weights initialization.
        """
        super().__init__()
        self.data_shape = data_shape
        self.data_size = torch.prod(torch.as_tensor(data_shape))
        self.weights_scale = weights_scale
        self.linear = nn.Linear(self.data_size, n_neurons)
        self.relu = nn.ReLU()

        self._initialize_weights()

    def forward(self, x):
        x_in = x
        x_lin = self.linear(x.flatten(start_dim=1))
        x = self.relu(x_lin)

        def hook(grad):
            self.hidden_grad = grad
        def hook_fn(grad):
            self.output_grad = grad  # Save gradient for further use
        
        x_lin.register_hook(hook)
        x.register_hook(hook_fn)
        output = x_in.flatten(start_dim=1) + x.mean(dim=1, keepdim=True)
        return output.unflatten(dim=1, sizes=self.data_shape) 
    
    def _initialize_weights(self):
        """
        Initialize the weights of the linear layer.
        """
        random_dir = torch.randn(1, self.linear.weight.shape[1])
        self.linear.weight = nn.Parameter(random_dir.repeat(self.linear.weight.shape[0], 1) * self.weights_scale)


def prepare_resnet(
        model_name,
        dataset_name,
        n_neurons,
        weights_scale, 
        bias_scale, 
        classification_weight_scale, 
        classification_bias_scale, 
        double_precision=False
        ):
    """Implement ResNet modification to launche the attack.
    Args:
        dataset_name(str): Name of the dataset to use. Available options are 'cifar10', 'cifar100', and 'imagenet'.
        n_neurons(int): Number of neurons in the malicious layer.
        weights_scale(float): Scale factor for the weights initialization.
        double_precision(bool): Whether to use double precision for the computations. Default is True.
    Returns:
    model(torch.nn.Module): Modified ResNet model.
    """

    if dataset_name not in ['cifar10', 'cifar100', 'imagenet']:
        raise ValueError("Invalid dataset name. Please, choose from 'cifar10', 'cifar100', and 'imagenet'.")
    if not model_name.lower().startswith('resnet'):
        raise ValueError("You are not using a ResNet model. Please, indicate a ResNet model.")
    
    if dataset_name == 'imagenet':
        model = getattr(torchvision.models, model_name)(weights=None)
        data_shape = (3, 224, 224)
    else:
        if dataset_name == 'cifar10':
             classes = 10
        else:
             classes = 100
             
        channels = 3
        data_shape = (3, 32, 32)
        if "-" in model_name.lower():  # Hacky way to separate ResNets from wide ResNets which are e.g. 28-10
                depth = int("".join(filter(str.isdigit, model_name.split("-")[0])))
                width = int("".join(filter(str.isdigit, model_name.split("-")[1])))
        else:
                depth = int("".join(filter(str.isdigit, model_name)))
                width = 1
                block, layers = resnet_depths_to_config(depth)
                model = ResNet(
                    block,
                    layers,
                    channels,
                    classes,
                    stem="CIFAR",
                    downsample="B",
                    width_per_group=(16 if len(layers) < 4 else 64) * width,
                    zero_init_residual=False,
                )

    def _initialize_malicious_weight(model, conv_weights_scale, conv_bias_scale, classification_weight_scale, classification_bias_scale):
        """Initialize resnet weights"""
        
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.Parameter(torch.randn(m.weight.shape) * conv_weights_scale)
                if m.bias is not None:
                    m.bias = nn.Parameter(m.bias * conv_bias_scale)
            elif isinstance(m, nn.Linear):
                class_weights = torch.randn(m.weight.shape[0], 1)
                m.weight = nn.Parameter(class_weights.repeat(1, m.weight.shape[1]) * classification_weight_scale)
                if m.bias is not None:
                    m.bias = nn.Parameter(torch.ones_like(m.bias) * classification_bias_scale)
            
            elif isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
        return model

    # def _initialize_malicious_weight(model, conv_weights_scale, conv_bias_scale, classification_weight_scale, classification_bias_scale):
    #     """Initialize resnet weights"""
        
    #     for m in model.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.constant_(m.weight, 0)
    #             if m.bias is not None:
    #                 m.bias = nn.Parameter(m.bias * conv_bias_scale)
    #         elif isinstance(m, nn.Linear):
    #             class_weights = torch.randn(m.weight.shape[0], 1)
    #             m.weight = nn.Parameter(class_weights.repeat(1, m.weight.shape[1]) * classification_weight_scale)
    #             if m.bias is not None:
    #                 m.bias = nn.Parameter(torch.ones_like(m.bias) * classification_bias_scale)
            
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.track_running_stats = False
    #     return model
    
    def _insert_malicious_block(model, n_neurons, weights_scale):
        """Insert malicious block in the model."""
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):  
                replacement = nn.Sequential(MaliciousBlock(data_shape=data_shape, n_neurons=n_neurons, weights_scale=weights_scale), module)
                if isinstance(module, nn.Conv2d):
                    nn.init.dirac_(module.weight)
                replace_module_by_instance(model, module, replacement)
                break
        return model
    
    model = _initialize_malicious_weight(model, weights_scale, bias_scale, classification_weight_scale, classification_bias_scale)
    model = _insert_malicious_block(model, n_neurons, weights_scale)

    if double_precision:
        model = model.double()
    return model

def replace_module_by_instance(model, old_module, replacement):
    def replace(model):
        for child_name, child in model.named_children():
            if child is old_module:
                setattr(model, child_name, replacement)
            else:
                replace(child)

    replace(model)
