import torch.nn as nn
import torchvision.models as models


class ChipPerformanceNet(nn.Module):
    """Predict chip performance from stacked TEM and AFM images."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        # Replace first conv to accept two-channel input
        self.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace classification head with regression output
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(1)
