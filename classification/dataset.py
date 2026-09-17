import json
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# paths from where to load the data
# please read datasets/README.md for more information about the dataset structure and where to place the datasets.

DEFAULT_TRAIN_CSV = "datasets/20250404-properties-dataset/CategorieTechnique_train.csv"
DEFAULT_VAL_CSV = "datasets/20250404-properties-dataset/CategorieTechnique_val.csv"
DEFAULT_IMAGES_DIR = "datasets/20250404-properties-dataset/images"
DEFAULT_CLASSES_JSON = (
    "datasets/20250404-properties-dataset/CategorieTechnique_classes.json"
)


# my idea was to implement a dataset class for each of the three classification tasks to make a cleaner code.


class CategorieTechniqueDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.images_dir, row["illustration"])
        image = Image.open(image_path).convert("RGB")
        label = int(row["label_ids"])

        if self.transform:
            image = self.transform(image)

        return image, label


class FormeFonctionDataset(Dataset):
    pass


class GenreDataset(Dataset):
    pass


def get_transforms_tiny():
    """
    From the doc of meta https://github.com/facebookresearch/ConvNeXt

    The model convnext_tiny is trained on ImageNet-1K with the following preprocessing:
    - Resize the image to 224x224 pixels.
    - Convert the image to a tensor.
    - Normalize the image using the mean and standard deviation of the ImageNet dataset.

    """

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_data(
    train_csv=DEFAULT_TRAIN_CSV,
    val_csv=DEFAULT_VAL_CSV,
    images_dir=DEFAULT_IMAGES_DIR,
    classes_json=DEFAULT_CLASSES_JSON,
    batch_size=32,
    max_train_samples=None,
    max_val_samples=None,
    num_workers=2,
):
    """
    Loads the training and validation datasets, applies transformations, and returns DataLoaders for both.
    """
    print("=" * 60)
    print("1. DATA LOADING")
    print("=" * 60)

    # reading csv
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    print(f"[Data] Train CSV : {len(df_train)} rows")
    print(f"[Data] Val CSV : {len(df_val)} rows")

    # Optional subsampling if specified (e.g. for quick tests)
    if max_train_samples is not None and max_train_samples < len(df_train):
        df_train = df_train.iloc[:max_train_samples]
        print(f"[Data] Using a train subset : {len(df_train)} rows")

    if max_val_samples is not None and max_val_samples < len(df_val):
        df_val = df_val.iloc[:max_val_samples]
        print(f"[Data] Using a val subset   : {len(df_val)} rows")

    # loading class names from the JSON file
    if os.path.exists(classes_json):
        with open(classes_json, "r", encoding="utf-8") as f:
            classes_dict = json.load(f)
            # we need to invert the dict to have {int: str} instead of {str: int}
            class_names = {int(v): k for k, v in classes_dict.items()}
    else:
        raise FileNotFoundError(f"File {classes_json} not found.")

    print(f"[Data] Detected classes ({len(class_names)}) : {class_names}")

    # applying transformations and creating datasets
    transform = get_transforms_tiny()  # TODO: automatically adapt the transform based on the model used (convnext_tiny, convnext_small, etc.)

    train_dataset = CategorieTechniqueDataset(df_train, images_dir, transform=transform)
    val_dataset = CategorieTechniqueDataset(df_val, images_dir, transform=transform)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,  # if using GPU, pin memory for faster data transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, class_names
