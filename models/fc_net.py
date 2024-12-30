import torch.nn as nn
import torch

class FCNet(nn.Module):
    """
    Fully connected neural network with ReLU activation functions."""
    def __init__(self, input_dimension, output_dimension, classification_weight_scale, hidden_layers=[1], initialization='equal', classification_bias=None,
                 input_weights_scale=1, hidden_weights_scale=1e-3, hidden_bias_scale=1e-3):
        """
        Args:
            input_dimension(int): Dimension of the input.
            output_dimension(int): Dimension of the output.
            hidden_layers(list): List of integers representing the number of neurons in each hidden layer. Deafult is [1].
            weight_scale(float): Scale of the weight of the first hidden layer of the network.
            initialization(str): Initialization method for the weights. Possible values are 'uniform', 'equal' and 'None'. Default is 'equal'.
            classification_bias(float): Bias magnitude for the classification layer. Default is None.
            input_weights_scale(float): Scale of the weights of the input layer. Default is 1.
            hidden_weights_scale(float): Scale of the weights of the hidden layers. Default is 1e-3.
            hidden_bias_scale(float): Scale of the bias of the hidden layers. Default is 1e-3.
        """
        super(FCNet, self).__init__()

        self.input_weights_scale = input_weights_scale
        self.classification_weight_scale = classification_weight_scale
        self.hidden_weights_scale = hidden_weights_scale
        self.hidden_bias_scale = hidden_bias_scale

        if hidden_layers is None:
            raise ValueError("You need to specify at least one hidden layer.")
        else:
            self.layers = nn.ModuleList([
                nn.Linear(input_dimension, hidden_layers[0]),
                nn.ReLU()
            ])
            self._initialize_att_weights(initialization)

            for i in range(1, len(hidden_layers)):
                self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                self.layers.append(nn.ReLU())
            
            self.layers.append(nn.Linear(hidden_layers[-1], output_dimension))
            if classification_bias is not None:
                self.layers[-1].bias.data = torch.ones_like(self.layers[-1].bias) * classification_bias
            
            if len(hidden_layers) > 1:
                self._initialize_hidden_layers()
            
            self._initialize_class_weights()

    
    def _initialize_hidden_layers(self):
        """
        Initialize the weights of the hidden layers.
        """
        for i in range(2, len(self.layers), 2):
            random_tensor = torch.randn(self.layers[i].weight.shape[0], 1)
            self.layers[i].weight.data = random_tensor.repeat(1, self.layers[i].weight.shape[1]) * self.hidden_weights_scale
            self.layers[i].bias.data = self.layers[i].bias.data * self.hidden_bias_scale

    def _initialize_class_weights(self):
        """
        Initialize the weights of the classification layer.
        """
        random_tensor = torch.randn(self.layers[-1].weight.shape[0], 1)
        self.layers[-1].weight.data = random_tensor.repeat(1, self.layers[-1].weight.shape[1]) * self.classification_weight_scale



    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

    def _initialize_att_weights(self, initialization):
        """
        Initialize the weights of the the first layer in the network.
        Args:
            initialization(str): Initialization method for the weights. Possible values are 'uniform', equal and 'None'. Default is 'equal'.
        """
        if initialization == 'uniform':
            return nn.init.uniform_(self.hidden_layers[0].weight, 0, 1 * self.self.input_weights_scale)
        elif initialization == 'equal':
            random_tensor = torch.randn(1, self.layers[0].weight.shape[1])
            self.layers[0].weight.data = random_tensor.repeat(self.layers[0].weight.shape[0], 1) * self.input_weights_scale
             
        else:
            raise NotImplementedError("Initialization method not implemented. ")
            
