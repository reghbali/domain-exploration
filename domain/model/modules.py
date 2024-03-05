from typing import Tuple

import torch.nn as nn
import numpy as np


class MLP_Block(nn.Module):
    """Building block for MLP-based models."""

    def __init__(
            self, hidden_size: int, activation: nn.Module, depth: int
    ) -> None:
        """Initialization of the MLP block.

        Args:
            hidden_size: Number of neurons in the linear layer.
            activation: Activation function.
            depth: Number of MLP blocks (linear layer with activation).
        """
        super(MLP_Block, self).__init__()
        layers = []
        for _ in range(depth):
            linear = nn.Linear(hidden_size, hidden_size)
            layers.append(linear)
            layers.append(activation)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Propagates the input through the MLP block.

        Args:
            x: Input.

        Returns:
            Output of the network.
        """
        return self.layers(x)


class MLP(nn.Module):
    def __init__(
            self,
            input_shape: Tuple[int],
            output_shape: Tuple[int],
            hidden_factor: int = 1,
            depth: int = 1,
    ) -> None:
        """Initialization of the multi-layer perceptron.

        Args:
            input_shape: Shape of the input.
            output_shape: Shape of the output.
            hidden_factor: Factor for multiplying with input length to
                determine the number of neurons in each hidden layer.
                Defaults to 1.
            depth: Number of hidden layers. Defaults to 1.
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        input_len = int(np.prod(input_shape))
        output_len = int(np.prod(output_shape))
        hidden_size = int(input_len * hidden_factor)

        self.layers = nn.ModuleList(
            [
                nn.Flatten(),
                nn.Linear(input_len, hidden_size),  # Input layer
                MLP_Block(hidden_size, nn.ReLU(), depth),
                nn.Linear(hidden_size, output_len),  # Output layer
            ]
        )

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """Propagates the input through the MLP block.

        Args:
            x: Input.

        Returns:
            Output of the network.
        """
        return self.layers(x)



class PatchEmbedding(nn.Module):
    def __init__(
            self, image_size: Tuple[int,int], patch_size: int, in_channels: int, embed_dim: int
    ) -> None:
        """Patch Embedding Layer 

        Args:
            image_size: Shape of the input image
            patch_size: Tunable parameter for patch size 
                        int: 64 -> [64,64]
            in_channels: Number of channels in input image
            embed_dim:  Tunable parameter
                        Defaults to (patch_size)*(patch_size)
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.flatten_size =  (patch_size) **2 * in_channels
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)


    def forward(self,x):
        """Embed Given Patch 

        Args:
            input: Patch 
                   shape = (b: Batch Size, c: Channels, h_p: Height, w_p: Width)
        """
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2) 
        return x



class TransformerEncoder(nn.Module):
    def __init__(
            self, embed_dim, num_heads,num_layers, dropout_rate
    ) -> None:
        """Encoder Layer

        Args:
            embed_dim: Shape of the input image
            num_heads: Number of MultiHeadAttention models
            num_layers: Number of channels in input image
            dropout_rate: Regulate to [0.1 - 0.29999...]  
        """
        
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, 
                                                   nhead=num_heads, 
                                                   dropout=dropout_rate)
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self,input):
        return self.encoder(input)

    
class VisionTransformer(nn.Module):
    def __init__(
            self, image_size, patch_size, in_channels, embed_dim, 
                  num_heads, num_layers, num_classes, 
                  dropout_rate=0.15
    ) -> None:
        """Vision Transformer Initialization

        Args: 
            {pre-defined}
        """
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, num_layers, dropout_rate)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self,x):
        """Propagates the input through the MLP block.

        Args:
            x: Input

        Returns:
            Output of Transformer
        """
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        cls_token = x[:, 0] 
        x = self.classifier(cls_token)
        return x