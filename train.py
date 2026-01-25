from src.data import build_dataset, get_dataset_root
from src.dataset import (
    dataSet,
    eval_transform,
    train_transform,
)
from src.model import Model
from src.train_utils import (
    set_backbone_trainable,
    build_criterion,
    build_optimizer,
    train_step,
    eval_step,
)
from pathlib import Path
from torch.utils.data import DataLoader


# Runs one full epoch
def train_one_epoch(model, loader, optimizer, criterion):
    loss_sum, acc_sum, n = 0.0, 0.0, 0

    for xb, yb in loader:
        loss, acc = train_step(
            model=model, images=xb, labels=yb, optimizer=optimizer, criterion=criterion
        )
        batch_size = xb.size(0)
        loss_sum += loss * batch_size
        acc_sum += acc * batch_size
        n += batch_size

    return loss_sum / n, acc_sum / n


# evaluates the data
def validate(model, loader, criterion):
    loss_sum, acc_sum, n = 0.0, 0.0, 0

    for xb, yb in loader:
        loss, acc = eval_step(model=model, images=xb, labels=yb, criterion=criterion)
        batch_size = xb.size(0)
        loss_sum += loss * batch_size
        acc_sum += acc * batch_size
        n += batch_size

    return loss_sum / n, acc_sum / n


def main():
    EPOCHS = 5
    LR = 1e-3
    WEIGHT_DECAY = 1e-2

    #! Change this line here to the name of your data folder (currently "data")
    DATA_DIR = Path.cwd() / "data"

    # Gets the location of the dataset images, determines the classes and gives each class an index
    # Returns a list of images and their labels
    dataset_root = get_dataset_root(DATA_DIR)
    train_data = build_dataset(dataset_root, "train")
    test_data = build_dataset(dataset_root, "test")

    # Converts the images and their labels into ResNet50 compatible tensors
    train_dataset = dataSet(train_data, train_transform)
    test_dataset = dataSet(test_data, eval_transform)

    # Setup dataloaders
    BATCH_SIZE = 64
    NUM_WORKERS = 4

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # randomly iterate the dataset
        num_workers=NUM_WORKERS,  # Number of processes simultaneously
        pin_memory=True,  # allows faster copying to GPU
        persistent_workers=(
            NUM_WORKERS > 0
        ),  # Only true if NumWorkers is greater than 0
    )
    # exact same as train_dataloader but without shuffling
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    # Training step preparations
    food_model = Model(num_classes=101)  # Create model
    set_backbone_trainable(
        food_model, requires_grad=False
    )  # Freeze ResNet50 so we only train our classifier

    # Building criterion and optimizer for the training loop
    criterion = build_criterion()
    optimizer = build_optimizer(model=food_model, lr=LR, weight_decay=WEIGHT_DECAY)

    # Training Loop
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            food_model,
            train_dataloader,
            optimizer,
            criterion,
        )
        val_loss, val_acc = validate(
            food_model,
            test_dataloader,
            criterion,
        )

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | "
            f"Eval Loss: {val_loss:.4f} | Eval Accuracy: {val_acc:.4f}"
        )


if __name__ == "__main__":
    main()
