"""
pipeline/report.py
------------------
Stage 5: Generate an HTML report from pipeline results using Jinja2.

Produces a self-contained HTML file that can be:
  - Opened directly in a browser
  - Served via the FastAPI /report endpoint
  - Printed to PDF via the browser's print dialog (or headless Chrome)

The template lives at templates/report.html.j2
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"


# ---------------------------------------------------------------------------
# Report data model
# ---------------------------------------------------------------------------

@dataclass
class PipelineReport:
    case_id:   str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Stage results (None = stage was skipped)
    lesion_detected:       Optional[bool]  = None
    lesion_confidence:     Optional[float] = None
    lesion_probabilities:  Optional[list]  = None

    segmentation_run:       bool           = False
    segmentation_mask_path: Optional[str]  = None
    lesion_volume_voxels:   Optional[int]  = None

    malignancy_label:       Optional[str]  = None   # "malignant" | "benign"
    malignancy_confidence:  Optional[float] = None
    malignancy_probabilities: Optional[list] = None

    # Embedded images (base64 PNG strings, filled by generate_figures)
    mip_figure_b64:  Optional[str] = None
    mask_figure_b64: Optional[str] = None

    error_message: Optional[str] = None

    @property
    def overall_impression(self) -> str:
        if self.error_message:
            return "Pipeline Error"
        if not self.lesion_detected:
            return "No lesion detected"
        if self.malignancy_label == "malignant":
            return "Malignant lesion detected"
        if self.malignancy_label == "benign":
            return "Benign lesion detected"
        return "Lesion detected (malignancy assessment not run)"


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def mip_channels_to_b64(mip_array: np.ndarray) -> str:
    """
    Render the 4 MIP channels as a 2×2 grid PNG and return as base64 string.
    mip_array: (4, 256, 256) float32
    """
    titles = ["Post1 − Pre", "Post2 − Pre", "Post2 − Post1", "Post2 (norm.)"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), facecolor="#1a1a2e")

    for ax, channel, title in zip(axes, mip_array, titles):
        ax.imshow(channel, cmap="gray", aspect="equal")
        ax.set_title(title, color="white", fontsize=9, pad=4)
        ax.axis("off")

    plt.tight_layout(pad=0.5)
    return _fig_to_b64(fig)


def mask_overlay_to_b64(mip_array: np.ndarray, mask_3d: np.ndarray) -> str:
    """
    Overlay the max-projected segmentation mask on the Post2 MIP channel.
    """
    post2_mip  = mip_array[3]                          # (256, 256)
    mask_2d    = (mask_3d > 0).max(axis=2).astype(float)  # (H, W)

    # Scale mask to MIP resolution if needed
    if mask_2d.shape != post2_mip.shape:
        import cv2
        mask_2d = cv2.resize(mask_2d, (post2_mip.shape[1], post2_mip.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor="#1a1a2e")
    axes[0].imshow(post2_mip,  cmap="gray");  axes[0].set_title("Post2 MIP",        color="white", fontsize=9); axes[0].axis("off")
    axes[1].imshow(post2_mip,  cmap="gray")
    axes[1].imshow(mask_2d,    cmap="Reds",   alpha=0.45)
    axes[1].set_title("Segmentation overlay", color="white", fontsize=9)
    axes[1].axis("off")

    plt.tight_layout(pad=0.5)
    return _fig_to_b64(fig)


def _fig_to_b64(fig: plt.Figure) -> str:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

def render_report(report: PipelineReport, output_path: str | Path | None = None) -> str:
    """
    Render the Jinja2 template with the report data and return the HTML string.
    If output_path is given, also write the HTML to that file.
    """
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )

    try:
        template = env.get_template("report.html.j2")
    except Exception as exc:
        logger.error(f"Could not load report template: {exc}")
        raise

    html = template.render(report=report)

    if output_path:
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"Report written → {output_path}")

    return html


# ---------------------------------------------------------------------------
# Convenience function used by orchestrator
# ---------------------------------------------------------------------------

def generate_report(
    report:     PipelineReport,
    mip_array:  np.ndarray | None = None,
    mask_3d:    np.ndarray | None = None,
    output_dir: str | Path | None = None,
) -> tuple[str, str]:
    """
    Generate figures, embed them in the report, render HTML.

    Returns
    -------
    (html_string, output_file_path_or_empty_string)
    """
    if mip_array is not None:
        try:
            report.mip_figure_b64 = mip_channels_to_b64(mip_array)
        except Exception as exc:
            logger.warning(f"Could not generate MIP figure: {exc}")

    if mip_array is not None and mask_3d is not None:
        try:
            report.mask_figure_b64 = mask_overlay_to_b64(mip_array, mask_3d)
        except Exception as exc:
            logger.warning(f"Could not generate mask overlay figure: {exc}")

    out_path = ""
    if output_dir:
        out_path = str(Path(output_dir) / f"{report.case_id}_report.html")

    html = render_report(report, output_path=out_path or None)
    return html, out_path
