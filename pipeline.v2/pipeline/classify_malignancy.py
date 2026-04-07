"""
pipeline/classify_malignancy.py
--------------------------------
Stage 4: Malignant vs Benign classification using EfficientNet.

Runs only when Stage 2 confirmed a lesion is present.
Optionally uses the nnUNet segmentation mask from Stage 3 to crop
the MIP to the lesion bounding box before classification — this
focuses the network on the region of interest rather than the full FOV.

Input  : (4, 256, 256) MIP array  +  optional segmentation mask
Output : MalignancyResult(label=0|1, confidence, probabilities)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

logger = logging.getLogger(__name__)

MALIGNANCY_MODEL_PATH = os.environ.get("MALIGNANCY_MODEL_PATH", "C:\\Programming\\SBME-Capstone-2025-Breast-Lesion-MRI-Analysis\\pipeline.v2\\weights\\3.31.mvb.best_model.pth")
NUM_CLASSES           = 2   # 0: benign, 1: malignant
IN_CHANNELS           = 4


# ---------------------------------------------------------------------------
# Model definition  (same architecture as lesion classifier, separate weights)
# ---------------------------------------------------------------------------

def build_malignancy_model(num_classes: int = NUM_CLASSES, in_channels: int = IN_CHANNELS) -> nn.Module:
    """
    EfficientNet-B0 adapted for 4-channel input and binary classification.
    Matches the training code from model_factory.py using efficientnet_pytorch.
    """
    # Don't download pretrained weights—we'll load from checkpoint
    model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)

    # Replace stem conv for 4 channels
    if in_channels != 3:
        old_conv = model._conv_stem
        model._conv_stem = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

    model._dropout = nn.Dropout(p=0.6)
    return model


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MalignancyResult:
    label: int                    # 0 = benign, 1 = malignant
    confidence: float
    probabilities: list[float]    # [p_benign, p_malignant]

    @property
    def is_malignant(self) -> bool:
        return self.label == 1

    @property
    def label_name(self) -> str:
        return "malignant" if self.is_malignant else "benign"

    def __str__(self) -> str:
        return f"{self.label_name.upper()}  (confidence {self.confidence:.1%})"


# ---------------------------------------------------------------------------
# ROI crop helper
# ---------------------------------------------------------------------------

def crop_mip_to_mask(
    mip_array: np.ndarray,
    mask_3d: np.ndarray,
    margin: int = 16,
) -> np.ndarray:
    """
    Project the 3-D segmentation mask to a 2-D bounding box (max-MIP axis),
    then crop each MIP channel to that bounding box + margin and resize back
    to (4, 256, 256).

    Parameters
    ----------
    mip_array : (4, 256, 256) — full-FOV MIP channels
    mask_3d   : (H, W, D)    — integer label volume from nnUNet
    margin    : pixels of padding around the lesion bounding box

    Returns
    -------
    (4, 256, 256) cropped and resized MIP array
    """
    import cv2

    # Project mask to 2-D (same axis as MIP generation = axis 2 → axis 0 of mip)
    mask_2d = (mask_3d > 0).max(axis=2).astype(np.uint8)  # (H, W)

    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)

    if not rows.any():
        logger.warning("Mask is empty — skipping ROI crop, using full MIP.")
        return mip_array

    rmin, rmax = int(rows.argmax()), int(len(rows) - rows[::-1].argmax() - 1)
    cmin, cmax = int(cols.argmax()), int(len(cols) - cols[::-1].argmax() - 1)

    # Scale bounding box from mask space to MIP space
    h_mask, w_mask = mask_2d.shape
    h_mip,  w_mip  = mip_array.shape[1], mip_array.shape[2]
    scale_r = h_mip / h_mask
    scale_c = w_mip / w_mask

    rmin_mip = max(0,     int(rmin * scale_r) - margin)
    rmax_mip = min(h_mip, int(rmax * scale_r) + margin)
    cmin_mip = max(0,     int(cmin * scale_c) - margin)
    cmax_mip = min(w_mip, int(cmax * scale_c) + margin)

    cropped_channels = []
    for ch in mip_array:
        crop = ch[rmin_mip:rmax_mip, cmin_mip:cmax_mip]
        resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
        cropped_channels.append(resized)

    return np.stack(cropped_channels, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class MalignancyClassifier:
    """
    Wrapper around the malignancy EfficientNet model.

    Usage
    -----
    clf    = MalignancyClassifier()
    result = clf.predict(mip_array)                        # full-FOV
    result = clf.predict(mip_array, mask_3d=seg_mask)      # ROI-cropped
    """

    def __init__(self, weights_path: str | Path | None = None) -> None:
        self.device = torch.device("cpu")
        self.model  = build_malignancy_model()

        path = Path(weights_path or MALIGNANCY_MODEL_PATH)
        if path.exists():
            checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                state = checkpoint["model_state"]
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
            else:
                state = checkpoint

            self.model.load_state_dict(state, strict=False)
            logger.info(f"Loaded malignancy model weights from {path}")
        else:
            logger.warning(
                f"Malignancy model weights not found at {path}. "
                "Running with random weights — for testing only."
            )

        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        mip_array: np.ndarray,
        mask_3d:   np.ndarray | None = None,
        margin:    int = 16,
    ) -> MalignancyResult:
        """
        Parameters
        ----------
        mip_array : (4, 256, 256) float32 MIP array
        mask_3d   : optional (H, W, D) segmentation mask from nnUNet;
                    if provided, the MIP is cropped to the lesion ROI first
        margin    : crop margin in MIP pixels (used only when mask_3d is given)
        """
        if mask_3d is not None:
            mip_array = crop_mip_to_mask(mip_array, mask_3d, margin=margin)

        if mip_array.ndim == 3:
            tensor = torch.from_numpy(mip_array).unsqueeze(0)
        else:
            tensor = torch.from_numpy(mip_array)

        tensor = tensor.to(self.device)
        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0)
        label  = int(probs.argmax().item())

        return MalignancyResult(
            label         = label,
            confidence    = float(probs[label].item()),
            probabilities = probs.tolist(),
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_classifier: MalignancyClassifier | None = None


def classify_malignancy(
    mip_array:    np.ndarray,
    mask_3d:      np.ndarray | None = None,
    weights_path: str | Path | None = None,
    margin:       int = 16,
) -> MalignancyResult:
    """
    Module-level entry point used by the pipeline orchestrator.
    Lazily initialises a singleton MalignancyClassifier on first call.
    """
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = MalignancyClassifier(weights_path)
    return _default_classifier.predict(mip_array, mask_3d=mask_3d, margin=margin)
