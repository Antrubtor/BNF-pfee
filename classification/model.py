from typing import cast

import torch
from torch import nn
from torchvision import models


def create_convnext_model(
    num_classes: int = 5,
    pretrained: bool = True,
    device: torch.device | str | None = None,
):
    """
    Creates a convnext model
    TODO: specify a model variant (tiny, small, base, large) and add a parameter for it
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)

    last_layer = cast(
        nn.Linear, model.classifier[2]
    )  # last layer is a Linear layer, we need to cast it to access its attributes

    model.classifier[2] = nn.Linear(last_layer.in_features, num_classes)

    model = model.to(device)
    print(
        f"[Model] ConvNeXt-Tiny  initialized with {num_classes} classes, pretrained={pretrained}, device={device}"
    )

    return model
