from pathlib import Path
import PIL 
import torch
import torchvision

# Takes the name of the folder that contains the food-101 dataset in it, and returns a full file path to allow for
# different file names on different devices
def get_dataset_root(data_dir):
    dataset_root = Path(data_dir) / "food-101"
    return dataset_root


# takes the path to the meta directory in food-101 and opens the classes.txt file, then proceeds to read each line and
# strip the "\n" character saving each food class as its own entry
def read_classes(meta_dir) -> list:
    classes_file = meta_dir / "classes.txt"
    with open(classes_file, "r") as file:
        lines = [line.strip() for line in file]
    return lines


# turns the list of class names to a dictionary where the key is the class name and the value is the index
# used later for classification on the final neuron layer
def class_to_index(classes_list):
    class_dict = {class_name: idx for idx, class_name in enumerate(classes_list)}
    return class_dict

