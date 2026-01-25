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


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


# A single epoch training step, runs forward pass, calculates loss, backpropogation, and makes a prediction per entry
def train_step(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)  # Stops gradient accumulation
    logits = model(images)  # forward pass
    loss = criterion(logits, labels)  # compute cross-entropy loss
    loss.backward()  # saves gradients into parameter.grad for each weight
    optimizer.step()  # adjusts weight depending on lr and parameter.grad
    acc = accuracy(logits.detach(), labels)  # calculates accuracy
    return loss.item(), float(acc)


def eval_step(
    model: nn.Module, images: torch.Tensor, labels: torch.Tensor, criterion: nn.Module
) -> tuple[float, float]:
    model.eval()

    # No gradient calculation
    with torch.no_grad():
        # same steps as training but nothing is changed or adjusted
        logits = model(images)
        loss = criterion(logits, labels)
        acc = accuracy(logits, labels)
        return loss.item(), float(acc)
