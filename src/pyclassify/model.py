from torch import flatten
import torch.nn as nn

class AlexNet(nn.Module):
    """
    This class implements Alexnet for image classification. More details regarding the architecture can be
    found at thw following link:
    https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

    Attributes:
        num_classes (int): The number classes that are considered.
        features (nn.Sequential): A nn.Sequential of layers, as specified in the article.
        avgpool (nn.AdaptiveAvgPool2d): Pooling layer.
        classifier (nn.Sequential): A nn.Sequential that is used for classification.

    Args:
        num_classes (int, optional): Number of output classes (10, unless otherwise specified)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: output, tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits
