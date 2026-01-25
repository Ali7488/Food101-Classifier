import torch
import torch.nn as nn


# Enables/Disables gradient computation for the model backbone, ensures we only train the classifier and not the ResNet-50
def set_backbone_trainable(model: nn.Module, requires_grad: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = requires_grad


# build and return loss function used in training
def build_criterion() -> nn.Module:
    return nn.CrossEntropyLoss()


# return a list of parameters that can be trained, aka requires_grad = true
def get_trainable_params(model: nn.Module) -> list[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


# Build and return optimizer for training, uses only trainable parameters so must be called after freezing/unfreezing
def build_optimizer(
    model: nn.Module, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        get_trainable_params(model), lr=lr, weight_decay=weight_decay
    )

