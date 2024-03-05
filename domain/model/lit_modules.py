from typing import Tuple, Dict, List, Any

import lightning as L
import torch
from torch.optim import lr_scheduler, Optimizer
from torchmetrics.functional.classification import multiclass_accuracy

from domain.model.modules import MLP


class LitClassificationModel(L.LightningModule):

    def __init__(
            self,
            net: str,
            num_classes: int
    ) -> None:
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300)
        return [optimizer], [scheduler]

    def infer_batch(
            self, batch: Dict[str, dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def test_step(self, batch, batch_idx):
        # this is the test loop
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        acc = multiclass_accuracy(y_hat, y, num_classes=self.num_classes)
        self.log('test_loss', loss, prog_bar=True)
        self.log('acc', acc, prog_bar=True)


class MLP_MNIST(LitClassificationModel):

    def __init__(self):
        super().__init__(net=MLP([28, 28], [10]), num_classes=10)
    

class MLP_CiFAR10(LitClassificationModel):
    def __init__(self):
        super().__init__(net=MLP([32, 32, 3], [10]), num_classes=10)
