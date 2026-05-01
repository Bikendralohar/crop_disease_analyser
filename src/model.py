"""
model.py
--------
Defines the EfficientNetB3 transfer learning model for plant disease classification.
Uses a two-phase approach:
  Phase 1 - Freeze backbone, train custom head only
  Phase 2 - Unfreeze deeper layers for fine-tuning
"""

import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


def build_model(num_classes, freeze_backbone=True):
    """
    Builds EfficientNetB3 with a custom classification head.

    Args:
        num_classes     : number of disease classes in the dataset
        freeze_backbone : if True, only the head is trainable (Phase 1)

    Returns:
        model (nn.Module)
    """
    # Load EfficientNetB3 pretrained on ImageNet
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        # Freeze all backbone weights — only head will be trained in Phase 1
        for param in model.parameters():
            param.requires_grad = False

    # Replace the default classifier head with a custom one
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),  # Regularization
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),               # Additional regularization
        nn.Linear(512, num_classes)       # Final output layer
    )

    return model


def unfreeze_backbone(model, unfreeze_from_layer=5):
    """
    Unfreezes deeper layers of the EfficientNet backbone for fine-tuning (Phase 2).
    Earlier layers (basic edges/textures) stay frozen; later layers are unlocked.

    Args:
        model               : the EfficientNetB3 model
        unfreeze_from_layer : index into model.features to start unfreezing from
    """
    layers = list(model.features.children())
    frozen_count   = 0
    unfrozen_count = 0

    for i, layer in enumerate(layers):
        if i >= unfreeze_from_layer:
            for param in layer.parameters():
                param.requires_grad = True
                unfrozen_count += 1
        else:
            frozen_count += 1

    print(f"[Model] Frozen layers: {frozen_count} | Unfrozen layers: {unfrozen_count}")


def count_parameters(model):
    """Prints total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params    : {total:,}")
    print(f"[Model] Trainable params: {trainable:,}")
