"""
pipeline/classify_lesion.py
---------------------------
Stage 2: Lesion vs No-Lesion classification using EfficientNet.

Input  : (4, 256, 256) float32 MIP array (from preprocess.compute_mip_array)
Output : LesionResult(label=0|1, confidence=float, probabilities=[p_no_lesion, p_lesion])

The model expects a 4-channel input tensor shaped (1, 4, 256, 256).
Weights path is configured via LESION_MODEL_PATH env var or passed directly.
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

LESION_MODEL_PATH = os.environ.get("LESION_MODEL_PATH", "C:\\Programming\\SBME-Capstone-2025-Breast-Lesion-MRI-Analysis\\pipeline.v2\\weights\\3.31.les.best_model.pth")
NUM_CLASSES       = 2   # 0: no_lesion, 1: lesion
IN_CHANNELS       = 4   # [MIP(post1-pre), MIP(post2-pre), MIP(post2-post1), MIP(post2)]


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_lesion_model(num_classes: int = NUM_CLASSES, in_channels: int = IN_CHANNELS) -> nn.Module:
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
class LesionResult:
    label: int                    # 0 = no_lesion, 1 = lesion
    confidence: float             # probability of the predicted class
    probabilities: list[float]    # [p_no_lesion, p_lesion]

    @property
    def lesion_detected(self) -> bool:
        return self.label == 1

    def __str__(self) -> str:
        verdict = "LESION DETECTED" if self.lesion_detected else "No lesion"
        return f"{verdict}  (confidence {self.confidence:.1%})"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class LesionClassifier:
    """
    Thin wrapper that loads model weights once and exposes a predict() method.

    Usage
    -----
    clf = LesionClassifier()                   # loads from LESION_MODEL_PATH
    clf = LesionClassifier("path/to/weights.pth")
    result = clf.predict(mip_array)            # mip_array: np.ndarray (4,256,256)
    """

    def __init__(self, weights_path: str | Path | None = None) -> None:
        self.device = torch.device("cpu")
        self.model  = build_lesion_model()

        path = Path(weights_path or LESION_MODEL_PATH)
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
            logger.info(f"Loaded lesion model weights from {path}")
        else:
            logger.warning(
                f"Lesion model weights not found at {path}. "
                "Running with random weights — for testing only."
            )

        self.model.eval()

    @torch.no_grad()
    def predict(self, mip_array: np.ndarray) -> LesionResult:
        """
        Parameters
        ----------
        mip_array : np.ndarray, shape (4, 256, 256), dtype float32
            Output of pipeline.preprocess.compute_mip_array()

        Returns
        -------
        LesionResult
        """
        if mip_array.ndim == 3:
            tensor = torch.from_numpy(mip_array).unsqueeze(0)  # (1, 4, 256, 256)
        else:
            tensor = torch.from_numpy(mip_array)

        tensor = tensor.to(self.device)
        logits = self.model(tensor)                             # (1, 2)
        probs  = torch.softmax(logits, dim=1).squeeze(0)       # (2,)
        label  = int(probs.argmax().item())

        return LesionResult(
            label         = label,
            confidence    = float(probs[label].item()),
            probabilities = probs.tolist(),
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_classifier: LesionClassifier | None = None


def classify_lesion(
    mip_array:    np.ndarray,
    weights_path: str | Path | None = None,
) -> LesionResult:
    """
    Module-level entry point used by the pipeline orchestrator.
    Lazily initialises a singleton LesionClassifier on first call.
    Pass weights_path to override the default on first use.
    """
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = LesionClassifier(weights_path)
    return _default_classifier.predict(mip_array)
