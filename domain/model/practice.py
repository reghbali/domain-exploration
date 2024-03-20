import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        num_classes: int,
        in_channels: int,
        conv1_feature_maps: int,
        conv2_feature_maps: int,
        kernel_size: int,
        pool_size: int,
        linear_size1: int,
        linear_size2: int,
        linear_size3: int,
    ) -> None:
        """Initialization of the multi-layer perceptron.

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
        """

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, conv1_feature_maps, kernel_size)
        self.conv2 = nn.Conv2d(conv1_feature_maps, conv2_feature_maps, kernel_size)

        self.pool = nn.MaxPool2d(pool_size, pool_size)

        conv_output_size = (image_size[0] - kernel_size + 1) // pool_size
        conv_output_size = (conv_output_size - kernel_size + 1) // pool_size
        conv_output_size = conv_output_size**2 * conv2_feature_maps

        self.dense1 = nn.Linear(conv_output_size, linear_size1)
        self.dense2 = nn.Linear(linear_size1, linear_size2)
        self.dense3 = nn.Linear(linear_size2, linear_size3)
        self.dense4 = nn.Linear(linear_size3, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = self.dense4(x)

        return x
