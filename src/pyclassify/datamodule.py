from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from torchvision import datasets, transforms


class CIFAR10DataModule(pl.LightningDataModule):
    """
    Lightning data module for CIFAR-10.

    Attributes:
        data_path (str): The directory where the dataset will be downloaded.
        batch_size (int): The batch size for data loaders.
        transform (torchvision.transforms.Compose): The set of transformations to apply to the dataset.
        train (torch.utils.data.Dataset): The training dataset after splitting.
        valid (torch.utils.data.Dataset): The validation dataset after splitting.
        test (torch.utils.data.Dataset): The testing dataset.

    Args:
        data_path (str, optional): Where to save the dataset (default is './data').
        batch_size (int, optional): The batch size for data loaders (default 64).
    """
    def __init__(self, data_path='./data', batch_size=64):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.RandomCrop((64, 64)),
            transforms.ToTensor()
        ])

    def prepare_data(self):
        """
        To prepare the datasets.
        """
        datasets.CIFAR10(root=self.data_path, download=True)

    def setup(self, stage=None):
        """
        Sets up the datasets.

        Args:
            stage (str, optional): actually unused in the code.
        """
        train = datasets.CIFAR10(
            root=self.data_path,
            train=True,
            transform=self.transform,
            download=False,
        )
        self.train, self.valid = random_split(train, lengths=[45000, 5000])
        self.test = datasets.CIFAR10(
            root=self.data_path,
            train=False,
            transform=self.transform,
            download=False,
        )

    def train_dataloader(self):
        """
        To create the dataloader used in training.

        Returns:
            DataLoader: dataloader for the training dataset.
        """
        train_loader = DataLoader(dataset=self.train, batch_size=self.batch_size, drop_last=True)
        return train_loader

    def val_dataloader(self):
        """
        Same as before, but for validation.

        Returns:
            DataLoader: dataloader for the validation dataset.
        """
        validation_loader = DataLoader(dataset=self.valid, batch_size=self.batch_size, drop_last=False)
        return validation_loader

    def test_dataloader(self):
        """
        Same as before, but for testing.

        Returns:
            DataLoader: dataloader for the test dataset.
        """
        test_loader = DataLoader(dataset=self.test, batch_size=self.batch_size, drop_last=False)
        return test_loader


# save all the checkpoints: automatically chosing the best one, no need to do it manually
# possible to specify the model
# also, integration with tensorboard! See the logging in lightning_logs
