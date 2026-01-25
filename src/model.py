import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class Model(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.num_classes = num_classes  # Declare number of classes were dealing with

        weights = (
            ResNet50_Weights.IMAGENET1K_V2
        )  #  Pre-trained ResNet50 weights, trained to know edges, shapes, textures and patterns

        self.backbone = models.resnet50(
            weights=weights
        )  # Load model with pre-trained weights

        in_features = (
            self.backbone.fc.in_features
        )  # gets feature vectors size before deleting classification layer

        self.backbone.fc = (
            nn.Identity()
        )  # Removes ImageNet classes, and ouputs feature vectors instead

        self.classifier = nn.Linear(
            in_features, num_classes
        )  # converts the feature into 101 weighted sums

    # Takes our images and turns them into raw scores (logits) to use later for softmax and cross-entropy loss
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_vector = self.backbone(
            images
        )  # Outputs tensor with dimensions (N, 2048), where N is the image, and 2048 is its feature vectors
        logits = self.classifier(
            feature_vector
        )  # turns the feature vectors into 101 scores, outputs (N, 101)
        return logits
