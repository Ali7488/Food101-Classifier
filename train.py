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


def main():
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
        shuffle=True, #randomly iterate the dataset
        num_workers=NUM_WORKERS, # Number of processes simultaneously
        pin_memory=True, # allows faster copying to GPU 
        persistent_workers=(NUM_WORKERS > 0), #Only true if NumWorkers is greater than 0
    )
    #exact same as train_dataloader but without shuffling
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )


if __name__ == "__main__":
    main()
