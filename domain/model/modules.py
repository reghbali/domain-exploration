from typing import Tuple

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class MLP_Block(nn.Module):
    """Building block for MLP-based models."""

    def __init__(self, hidden_size: int, activation: nn.Module, depth: int) -> None:
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
            self,
            image_size: Tuple[int, int],
            patch_size: int,
            in_channels: int,
            embed_dim: int,
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
        self.flatten_size = (patch_size) ** 2 * in_channels
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
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
            self, 
            embed_dim: int, 
            num_heads: int, 
            num_layers: int, 
            dropout_rate
    ) -> None:
        """Encoder Layer

        Args:
            embed_dim: Shape of the input image
            num_heads: Number of MultiHeadAttention models
            num_layers: Number of channels in input image
            dropout_rate: Regulate to [0.1 - 0.29999...]
        """

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input):
        return self.encoder(input)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        dropout_rate=0.15,
    ) -> None:
        """Vision Transformer Initialization

        Args:
            {pre-defined}
        """
        super().__init__()

        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)

        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, num_layers, dropout_rate)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """Propagates the input through the Transformer

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


class CNN(nn.Module):
    def __init__(
            self, 
            input_shape: tuple[int,int],
            num_classes: int, 
            in_channels: int,
            conv1_feature_maps: int, 
            conv2_feature_maps: int,
            kernel_size: int,
            pool_size: int,  
            linear_size1: int, 
            linear_size2: int,
            linear_size3: int,
            activation = nn.functional.gelu
    ) -> None:
        """Initialization of the Convolutional Neural Network.

        Args:
            image_size: Shape of the input
            num_classes: Classes 
            in_channels: Number of channels in input image
            conv1_feature_maps: Number of filters conv1
            conv2_feature_maps: Number of filters conv2
            kernel_size: Filter size
            pool_size: Size of pool
            linear_size1: 
            linear_size2:
            linear_size3:  
            activation: default geLu, other (nn.Module) activations accepted
        """
        super().__init__()
        self.image_size
        output_shape = (input_shape[0] - kernel_size + 1) // pool_size
        output_shape = (output_shape - kernel_size + 1) // pool_size
        output_shape = output_shape ** 2 * conv2_feature_maps

        self.layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels, conv1_feature_maps, kernel_size),
                nn.MaxPool2d(pool_size, pool_size),
                activation(),

                nn.Conv2d(conv1_feature_maps, conv2_feature_maps, kernel_size),
                nn.MaxPool2d(pool_size, pool_size),
                activation(),
                
                nn.Flatten(), 
                nn.Linear(output_shape, linear_size1),
                activation(),
                nn.Linear(linear_size1, linear_size2),
                activation(), 
                nn.Linear(linear_size2, linear_size3),
                activation(),
                nn.Linear(linear_size3, num_classes)  
            ]
        )

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class ResCNN_block(nn.Module):
    def __init__(
            self, 
            in_channels, 
            intermediate_channels, 
            identity_downsample = None, 
            stride = 1, 
            activation= nn.ReLU(),
            normalization = nn.InstanceNorm2d
    ) -> None:
        """Initialization of the Convolutional Block used for residuality

        Args:
            in_channels: Number of channels in input image
            identity_downsample: Used if we need to change the shape
            activation: default ReLU, other torch.functional activations accepted
            expansion: Number of channels after it enters a ResCNN_block is 4x what it entered as 

        """
        super().__init__()
        self.expansion = 4
        self.activation = activation
        self.identity_downsample = identity_downsample
        self.stride = stride
    
        self.conv_1 = nn.Conv2d(in_channels,intermediate_channels,kernel_size=1,stride=1,padding=0,bias=False,)
        self.norm_1 = normalization(intermediate_channels)
        self.conv_2 = nn.Conv2d(intermediate_channels,intermediate_channels,kernel_size=3,stride=stride,padding=1,bias=False,)
        self.norm_2 = normalization(intermediate_channels)
        self.conv_3 = nn.Conv2d(intermediate_channels,intermediate_channels * self.expansion,kernel_size=1,stride=1,padding=0,bias=False,)
        self.norm_3 = normalization(intermediate_channels * self.expansion)


    def forward(self, x):
        identity = x.clone()

        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.activation(x)
        x = self.conv_3(x)
        x = self.norm_3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity

        x = self.activation(x)

        return x


class ResCNN(nn.Module):
    def __init__(
            self, 
            ResCNN_block,
            layers: tuple[int, int, int, int] = (3, 4, 6, 3), 
            in_channels: int = 3,
            num_classes: int = 1000,                
            activation = nn.ReLU()
    ) -> None: 
        """Initialization of the Residual CNN

        Args:
            layers: Repitition of each residual block (Default ResNet50 -- [3,4,6,3])
            in_channels: Number of channels in input image
            num_classes: Classes
            activation: default ReLU, other torch.functional activations accepted

                
        """  
       
        super(ResCNN,self).__init__()
        self.in_channels = 64
        self.conv_0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm_0 = nn.BatchNorm2d(64)
        self.activation = activation
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.initialize_layer(ResCNN_block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.initialize_layer(ResCNN_block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.initialize_layer(ResCNN_block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.initialize_layer(ResCNN_block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dense = nn.Linear(512 * 4, num_classes)
    
    def forward(self, x):
        x = self.conv_0(x)
        x = self.norm_0(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dense(x)

        return x

    def initialize_layer(self, ResCNN_block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            """
            This step adapts the identity(skip connection) so that its able to be added to proceeding layers
                (a) half input space [56x56 -> 28x28 (stride=2)]
                (b) or channels change
            """
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,intermediate_channels * 4,kernel_size=1,stride=stride,bias=False,),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(ResCNN_block(self.in_channels, intermediate_channels, identity_downsample, stride))

        self.in_channels = intermediate_channels * 4
        
        for i in range(num_residual_blocks - 1):
            layers.append(ResCNN_block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


class MLP_Block_Residual(nn.Module):
    def __init__(
            self, 
            input_size: int, 
            output_size: int, 
            activation= nn.ReLU()
    ) -> None:
        """Initialization of the Residual MLP Block

        Args:
            input_size: Size of input image
            output_size:  Size of output imgage
            activation: default ReLU, other torch.functional activations accepted
        """  
        super(MLP_Block_Residual, self).__init__()
        self.activation = activation
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(output_size, output_size)
        self.adjust_dimensions = None
        
        if input_size != output_size:
            self.adjust_dimensions = nn.Linear(input_size, output_size)

    def forward(self, x):
        identity = x
        
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        
        if self.adjust_dimensions is not None:
            identity = self.adjust_dimensions(identity)
        
        x += identity
        x = self.activation(x)
        return x


class ResidualMLP(nn.Module):
    def __init__(
            self, 
            input_shape, 
            layers: tuple[int], 
            num_classes: int, 
            activation=nn.GELU()
    ) -> None:
        """Initialization of the Residual multi-layer perceptron.

        Args:
            input_shape: Size of input image
            layers: Number of residual layers to be added
            num_classes: Classes in dataset
            activation: default GELU, other torch.functional activations accepted
                
        """  
        super(ResidualMLP, self).__init__()
        self.activation = activation
        input_size = input_shape[0] * input_shape[1] * input_shape[2]
        self.layers = nn.ModuleList(
            [
                nn.Flatten(),
                nn.Linear(input_size, layers[0]),
            ]
        )
        
        for i in range(1, len(layers)): 
            self.layers.append(MLP_Block_Residual(layers[i-1], layers[i], self.activation))
        
        self.layers.append(nn.Linear(layers[-1], num_classes))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)
    