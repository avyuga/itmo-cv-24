import torch.nn as nn
import torchvision

from torchvision.models import efficientnet


def build_simple_model():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding="same"), # (N, 16, 200, 200)
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding="same"), # (N, 32, 200, 200)
        nn.MaxPool2d(kernel_size=2), # (N, 32, 100, 100)

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"), # (N, 64, 100, 100)
        nn.MaxPool2d(kernel_size=2), # (N, 64, 50, 50)

        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding="same"), # (N, 32, 50, 50)
        nn.MaxPool2d(kernel_size=2), # (N, 32, 25, 25)
        nn.ReLU(),

        nn.Flatten(),
        nn.Linear(in_features=32*25*25, out_features=128),
        nn.Linear(in_features=128, out_features=5),
        nn.Softmax(dim=1)
 )


def build_efficientnet_b0(n_classes):
    base_model = efficientnet.efficientnet_b0(weights=efficientnet.EfficientNet_B0_Weights.IMAGENET1K_V1)
    base_model.classifier = nn.Sequential(
        nn.Linear(base_model.classifier[1].in_features, n_classes),
        nn.Softmax(dim=1)
    )
    return base_model


def build_efficientnet_b4(n_classes):
    base_model = efficientnet.efficientnet_b4(weights=efficientnet.EfficientNet_B4_Weights.IMAGENET1K_V1)
    base_model.classifier = nn.Sequential(
        nn.Linear(base_model.classifier[1].in_features, n_classes),
        nn.Softmax(dim=1)
    )
    return base_model