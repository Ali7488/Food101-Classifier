from pathlib import Path


# Takes the name of the folder that contains the food-101 dataset in it, and returns a full file path to allow for
# different file names on different devices
def get_dataset_root(data_dir: Path) -> Path:
    dataset_root = Path(data_dir) / "food-101"
    return dataset_root

# getter methods for the directories of meta files and image files
def get_meta_dir(dataset_root: Path) -> Path:
    return dataset_root / "meta"


def get_image_dir(dataset_root: Path) -> Path:
    return dataset_root / "images"

# takes the path to the meta directory in food-101 and opens the classes.txt file, then proceeds to read each line and
# strip the " " character saving each food class as its own entry
def read_classes(meta_dir: Path) -> list[str]:
    classes_file = meta_dir / "classes.txt"
    with open(classes_file, "r") as file:
        lines = [line.strip() for line in file]
    return lines

# turns the list of class names to a dictionary where the key is the class name and the value is the index
# used later for classification on the final neuron layer
def class_to_index(classes_list: list) -> dict[str, int]:
    class_dict = {class_name: idx for idx, class_name in enumerate(classes_list)}
    return class_dict

def split_data_labels(meta_dir: Path, image_dir: Path, classes_dict: dict[str, int], split: str) -> list[tuple[Path, int]]:
    if split == "train":
        split_file = meta_dir / "train.txt"
    elif split == "test":
        split_file = meta_dir / "test.txt"
    else:
        raise ValueError("""split must be either "train" or "test" """)

    with open(split_file, "r") as file:
        list_of_paths: list[tuple[Path, int]] = []
        for line in file:
            rel_path = Path(line.strip())
            class_name = rel_path.parent.name
            image_id = rel_path.name
            image_path = image_dir / class_name / f"{image_id}.jpg"
            label = classes_dict[class_name]
            list_of_paths.append((image_path, label))
        return list_of_paths

def validate_dataset(dataset_root: Path) -> None:
    if not dataset_root.exists():
        raise FileNotFoundError("Dataset Root not found")

    meta_dir = get_meta_dir(dataset_root)
    image_dir = get_image_dir(dataset_root)

    if not meta_dir.exists():
        raise FileNotFoundError("Meta directory not found")
    if not image_dir.exists():
        raise FileNotFoundError("Images directory not found")

    if not (meta_dir / "classes.txt").exists():
        raise FileNotFoundError("classes.txt not found")

    if not (meta_dir / "train.txt").exists():
        raise FileNotFoundError("train.txt not found")

    if not (meta_dir / "test.txt").exists():
        raise FileNotFoundError("test.txt not found")
    



