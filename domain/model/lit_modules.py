from typing import Tuple, Dict

import lightning as L
import torch
from torch.optim import lr_scheduler, Optimizer
from torchmetrics.functional.classification import multiclass_accuracy

from domain.model.modules import MLP, ResidualMLP, ResCNN, ResCNN_block, VisionTransformer 


class LitClassificationModel(L.LightningModule):
    def __init__(self, net: str, num_classes: int) -> None:
        """Initialization of the custom Lightning Module.

        Args:
            model: Neural network model name.
            config: Neural network model and training config.
        """
        super().__init__()
        self.num_classes = num_classes
        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()

    def configure_optimizers(
        self,
    ) -> Tuple[Optimizer, lr_scheduler.LRScheduler]:
        """Configures the optimizer and scheduler based on the learning rate
            and step size.

        Returns:
            Configured optimizer and scheduler.
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300)
        return [optimizer], [scheduler]

    def infer_batch(self, batch: Dict[str, dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate given batch through the Lightning Module.

        Args:
            batch: Batch containing the subjects.

        Returns:
            Model output and corresponding ground truth.
        """
        x, y = batch
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch: Dict[str, dict], batch_idx: int) -> float:
        """Infer batch on training data, log metrics and retrieve loss.

        Args:
            batch: Batch containing the subjects.
            batch_idx: Number displaying index of this batch.

        Returns:
            Calculated loss.
        """
        y_hat, y = self.infer_batch(batch)

        # Calculate loss
        loss = self.criterion(y_hat, y)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        acc = multiclass_accuracy(y_hat, y, num_classes=self.num_classes)
        return loss, acc

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        self.log('acc', acc, prog_bar=True)
        return {'val_acc': acc, 'val_loss': loss}

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        self.log('acc', acc, prog_bar=True)
        return {'test_acc': acc, 'test_loss': loss}


class MLP_MNIST(LitClassificationModel):
    def __init__(self):
        super().__init__(net=MLP([28, 28], 10), num_classes=10)


class MLP_CIFAR10(LitClassificationModel):
    def __init__(self):
        super().__init__(net=MLP([32, 32, 3], 10), num_classes=10)


class MLP_ImageNet(LitClassificationModel):
    def __init__(self):
        super().__init__(net=MLP([224, 224, 3], 1000), num_classes=1000)


class ResidualMLP_MNIST(LitClassificationModel):
    def __init__(self):
        super().__init__(
            net=ResidualMLP(
                input_shape=(1, 28, 28),
                num_classes=10,
                layers=[128, 128]
            ),
            num_classes=10
        )


class ResidualMLP_CIFAR10(LitClassificationModel):
    def __init__(self):
        super().__init__(
            net=ResidualMLP(
                input_shape=(3, 32, 32),
                num_classes=10,
                layers=[256, 256],
                hidden_factor=4
            ),
            num_classes=10
        )


class ResidualMLP_ImageNet(LitClassificationModel):
    def __init__(self):
        super().__init__(
            net=ResidualMLP(
                input_shape=(3, 224, 224),
                num_classes=1000,
                layers=[512, 512],
                hidden_factor=4
            ),
            num_classes=1000
        )


class CNN_MNIST(LitClassificationModel):
    def __init__(self):
        super().__init__(
            net=CNN(
                input_shape=(28, 28),
                num_classes=10,
                in_channels=1,
                conv1_feature_maps=32,
                conv2_feature_maps=64,
                kernel_size=3,
                pool_size=2,
                linear_size1=128,
                linear_size2=64,
                linear_size3=32,
                activation=nn.ReLU,
            ),
            num_classes=10,
        )


class CNN_CIFAR10(LitClassificationModel):
    def __init__(self):
        super().__init__(
            net=CNN(
                input_shape=(32, 32),
                num_classes=10,
                in_channels=3,
                conv1_feature_maps=32,
                conv2_feature_maps=64,
                kernel_size=3,
                pool_size=2,
                linear_size1=128,
                linear_size2=64,
                linear_size3=32,
                activation=nn.ReLU,
            ),
            num_classes=10,
        )


class CNN_ImageNet(LitClassificationModel):
    def __init__(self):
        super().__init__(
            net=CNN(
                input_shape=(224, 224),
                num_classes=1000,
                in_channels=3,
                conv1_feature_maps=64,
                conv2_feature_maps=128,
                kernel_size=3,
                pool_size=2,
                linear_size1=512,
                linear_size2=256,
                linear_size3=128,
                activation=nn.ReLU,
            ),
            num_classes=1000,
        )


class ResCNN_MNIST(LitClassificationModel):
    def __init__(self):
        super().__init__(
            net=ResCNN(
                ResCNN_block=ResCNN_block,
                layers=(2, 2, 2, 2),
                in_channels=1,
                num_classes=10,
                activation=nn.ReLU(),
            ),
            num_classes=10
        )


class ResCNN_CIFAR10(LitClassificationModel):
    def __init__(self):
        super().__init__(
            net=ResCNN(
                ResCNN_block=ResCNN_block,
                layers=(3, 4, 6, 3),
                in_channels=3,
                num_classes=10,
                activation=nn.ReLU(),
            ),
            num_classes=10
        )


class ResCNN_ImageNet(LitClassificationModel):
    def __init__(self):
        super().__init__(
            net=ResCNN(
                ResCNN_block=ResCNN_block,
                layers=(3, 4, 6, 3),  # resnet50 config
                in_channels=3,
                num_classes=1000,
                activation=nn.ReLU(),
            ),
            num_classes=1000
        )


class VisionTransformer_CIFAR10(LitClassificationModel):
    def __init__(self):
        super().__init__(
            net=VisionTransformer(
                input_shape=(3, 32, 32),
                in_channels=3, 
                num_classes=10, 
                dropout_rate=0.1, 
                patch_size=8, 
                embed_dim=256, 
                num_attention_heads=8, 
                num_layers=8, 
                attention_head_dim=32, 
                mlp_head_dim=1024, 
                freq=False
            ),
            num_classes=10
        )


class VisionTransformer_MNIST(LitClassificationModel):
    def __init__(self):
        super().__init__(
            net=VisionTransformer(
                input_shape=(1, 28, 28),
                in_channels=1,
                num_classes=10,
                dropout_rate=0.1,
                patch_size=7,
                embed_dim=256,
                num_attention_heads=8,
                num_layers=8,
                attention_head_dim=32,
                mlp_head_dim=1024,
                freq=False
            ),
            num_classes=10
        )


class VisionTransformer_ImageNet(LitClassificationModel):
    def __init__(self):
        super().__init__(
            net=VisionTransformer(
                input_shape=(3, 224, 224),
                in_channels=3,
                num_classes=1000,
                dropout_rate=0.1,
                patch_size=16,  # standard ViT imagenet patch_size
                embed_dim=768,
                num_attention_heads=12,
                num_layers=12,
                attention_head_dim=64,
                mlp_head_dim=3072,
                freq=False
            ),
            num_classes=1000
        )