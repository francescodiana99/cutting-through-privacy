"""This file is a basic CNN implementation inspired from https://ieeexplore.ieee.org/document/10190537"""

import torch
import torch.nn as nn

class ConvNet(nn.Module):
    """CNN model without pooling and BN."""

    def __init__(
            self, 
            num_classes=10, 
            in_channels=3, 
            n_attack_neurons=1000, 
            classification_weight_scale=1e-3,
            classification_bias_scale=1e12,
            attack_weight_scale=1e-3,
            input_dim=  32 * 32 * 3
            ):
        super(ConvNet, self).__init__()
        self.classification_weight_scale = classification_weight_scale
        self.classification_bias_scale = classification_bias_scale
        self.attack_weight_scale = attack_weight_scale
        self.input_dim = input_dim
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
        )
        self.dense_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, n_attack_neurons),
            nn.ReLU(),
            nn.Linear(n_attack_neurons, num_classes)
        )
        self._initialize_malicious_weights()

    def forward(self, x):
        x_input = x.detach()
        x = self.conv_block(x)
        x = x[:, :3, :, :]
        x = torch.flatten(x, start_dim=1)

        x = self.dense_block(x)
        
        return x
    
    def _initialize_malicious_weights(self):
        """Weight initialization for the attack."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.dirac_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        random_tensor = torch.randn(1, self.dense_block[1].weight.shape[1])
        self.dense_block[1].weight = nn.Parameter(random_tensor.repeat( self.dense_block[1].weight.shape[0], 1) * self.attack_weight_scale)

        random_class_weight = torch.randn(self.dense_block[3].weight.shape[0], 1)
        self.dense_block[3].weight = nn.Parameter(random_class_weight.repeat(1, self.dense_block[3].weight.shape[1]) * self.classification_weight_scale)
        self.dense_block[3].bias = nn.Parameter(torch.ones_like(self.dense_block[3].bias) * self.classification_bias_scale)


if __name__ == "__main__":
    model = ConvNet()
    print(model)
    x = torch.rand(2, 3, 32, 32)
    y = model(x)
    print(model)
    