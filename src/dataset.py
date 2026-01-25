from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Show images as is for evaluation
eval_transform = transforms.Compose(
    [
        transforms.Resize(
            256
        ),  # Resize shortest side to 256 and maintains aspect ratio
        transforms.CenterCrop(224),  # Center crop to 224x224
        transforms.ToTensor(),  # Convert from PIL to tensor float
        # Normalize to ImageNet statistics
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

# show pictures in slightly different ways to aid learning
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),  # Random cropping and resizing to 224x224
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        transforms.ToTensor(),  # Convert from PIL to float Tensor
        # Normalize to ImageNet statistics
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


class dataSet(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples # list of paths and their class_idx
        self.transform = transform # converter from PIL to tensor

        if self.transform is None:
            raise ValueError(
                "Dataset requires a please choose eval_transform or train_transform"
            )

    # return length of samples
    def __len__(self):
        return len(self.samples)

    # turn a path into a PIL image
    def __getitem__(self, index):
        image_path, label = self.samples[index]  # unpack the tuple at samples[index]
        image = Image.open(image_path).convert("RGB")  # open PIL image

        # transform to tensors and return
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor
