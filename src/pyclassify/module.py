import torch
import torchmetrics
import lightning.pytorch as pl
import torch.nn as nn

class Classifier(pl.LightningModule):
    """
    Lightning wrapper. Useful for streamlining operations related to training, validation and testing.

    Attributes:
        model (nn.Module): The model used
        train_accuracy (torchmetrics.Accuracy): Metric used during training.
        val_accuracy (torchmetrics.Accuracy): Metric used during validation.
        test_accuracy (torchmetrics.Accuracy): Metric used during testing.

    Args:
        model (nn.Module): The model to be used.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.model.num_classes)

    def _classifier_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        loss = torch.nn.functional.cross_entropy(logits, true_labels)
        pred_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, pred_labels
    
    def training_step(self, batch, _):
        """
        To perform a training step.

        Args:
            batch (tuple): Tuple containing input features and their labels.

        Returns:
            torch.Tensor: Loss computed for this training step.
        """
        loss, true_labels, pred_labels = self._classifier_step(batch)
        self.train_accuracy(pred_labels, true_labels)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_accuracy, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        """
        To perform a validation step.
        Args:
            batch (tuple): Tuple containing input features and their labels.
        """
        loss, true_labels, predicted_labels = self._classifier_step(batch)
        self.val_accuracy(predicted_labels, true_labels)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_accuracy, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, _):
        """
        To perform a test step.

        Args:
            batch (tuple): Tuple containing input features and their labels.
        """
        loss, true_labels, pred_labels = self._classifier_step(batch)
        self.test_accuracy(pred_labels, true_labels)
        self.log("test_acc", self.test_accuracy, on_epoch=True, on_step=False)

    def forward(self, x):
        """
        Just defining the forward pass by using the one of the model.
        """
        return self.model(x)
    
    def configure_optimizers(self):
        """
        As the name suggests, used to configure the optimizer.
        
        Returns:
        torch.optim.Adam: The optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer


# Some remarks:
# in trainer enable progress bar set to false
# strategy: auto. if it finds more than 1 gpu, it will go directly in parallel. bu ddp is not the only strategy!
# use ddp if the model fits in the GPU.
# Else you cannot use it. Use other stuff, but need to change a lot the lightning module
# (eg when 200M parameters). Also depending on the GPU. Surely e.g. 1B parameters -> need to parallelize
# you can also access logs while training!
# Remark that this one here is not a good training. fixed learning rate, ..., scheduler...
# you can access at runtime!
# when running out of time, lightning knows, at 99% it starts saving the checkpoint at each epoch
# you can also pretrain starting from an already pretrained model
# in config.yaml pass 'ckpt_path' ---> it will restart the training from that checkpoint
# create a trainer that loads the checkpoint for the solver.
# Max epoch -1 = 1000
# epoch=(eg)34 in the checkpoint, train until 1000.
# Also flag for restarting, to find the best batch and so on and so forth, or accumulate gradients + single
# backward pass...
# For this exercise remember to push the data so that when accessing from slurm you recover them
