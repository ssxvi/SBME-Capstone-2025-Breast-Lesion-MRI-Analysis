#!/usr/bin/env python3
"""
Overlay ISPY1 manual masks onto corresponding bias-corrected NIfTI images and export PNGs.

Default assumptions:
- Images: ispy1/images_bias-corrected_nifti/ISPY1_XXXX/ISPY1_XXXX_DCE_000{0,1,2}_N3*.nii.gz
  (also supports z-scored/resampled variants like *_N3_zscored.nii.gz)
- Masks:  ispy1/stv_manual_masks/ISPY1_XXXX.nii.gz
- Preferred timepoint order: 0002 -> 0001 -> 0000

Usage examples:
  python "Conversion Tools/overlay_masks.py"
  python "Conversion Tools/overlay_masks.py" --subjects 1147 1130 --num-slices 9
  python "Conversion Tools/overlay_masks.py" --all --alpha 0.6 --outline
  python "Conversion Tools/overlay_masks.py" --images-root d:/data/ispy1/images_bias-corrected_nifti --masks-root d:/data/ispy1/stv_manual_masks --out-dir d:/data/ispy1/overlays

Viewer examples:
  # Create an interactive HTML viewer per subject (slider across slices)
  python "Conversion Tools/overlay_masks.py" --viewer
  # Create viewer for given subjects using the saved overlays
  python "Conversion Tools/overlay_masks.py" --subjects 1147 1130 --viewer
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Iterable, List, Optional, Sequence, Tuple, Dict
import base64

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to
from nibabel.orientations import aff2axcodes
import csv


def find_repo_root() -> str:
    # Anchor defaults relative to this script, assuming repo root is two directories up
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    return repo_root


def list_subject_ids(images_root: str) -> List[str]:
    subject_ids: List[str] = []
    if not os.path.isdir(images_root):
        return subject_ids
    for name in os.listdir(images_root):
        if name.startswith("ISPY1_") and os.path.isdir(os.path.join(images_root, name)):
            # e.g., ISPY1_1147
            subject_ids.append(name.split("_", 1)[1])
    subject_ids.sort()
    return subject_ids


def choose_timepoint_file(subject_dir: str, preferred: Sequence[str]) -> Optional[str]:
    """
    Pick NIfTI volume file in preferred DCE order (e.g., 0002, 0001, 0000).
    """
    if not os.path.isdir(subject_dir):
        return None
    files = [f for f in os.listdir(subject_dir) if f.lower().endswith(".nii.gz")]
    if not files:
        return None
    # Build mapping from timepoint -> filename
    # Support filenames like ...DCE_0002_N3.nii.gz and ...DCE_0002_N3_zscored.nii.gz
    tp_regex = re.compile(r"DCE_(\d{4}).*\.nii\.gz$", re.IGNORECASE)
    tp_to_file = {}
    for f in files:
        m = tp_regex.search(f)
        if m:
            tp_to_file[m.group(1)] = f
    for tp in preferred:
        if tp in tp_to_file:
            return os.path.join(subject_dir, tp_to_file[tp])
    # Fallback to first .nii.gz if preferred not found
    return os.path.join(subject_dir, files[0]) if files else None


def load_nifti_img(path: str) -> nib.spatialimages.SpatialImage:
    return nib.load(path)


def resample_mask_to_image(mask_img: nib.spatialimages.SpatialImage, image_img: nib.spatialimages.SpatialImage) -> nib.spatialimages.SpatialImage:
    """
    Resample mask to the reference image grid using nearest-neighbor interpolation.
    This handles size, spacing, orientation, and affine differences.
    """
    try:
        resampled = resample_from_to(mask_img, image_img, order=0)  # nearest-neighbor preserves labels
        return resampled
    except Exception as e:
        raise RuntimeError(f"Failed to resample mask to image space: {e}")


def window_image(image: np.ndarray, lower_percentile: float = 5.0, upper_percentile: float = 95.0) -> np.ndarray:
    """
    Robust intensity windowing to [0,1].
    """
    finite_vals = image[np.isfinite(image)]
    if finite_vals.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    vmin = np.percentile(finite_vals, lower_percentile)
    vmax = np.percentile(finite_vals, upper_percentile)
    if vmax <= vmin:
        vmax = vmin + 1.0
    img_w = np.clip((image - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)
    return img_w


def find_mask_slices(mask: np.ndarray, max_slices: int) -> List[int]:
    """
    Return slice indices with non-zero mask, prioritized by mask area, capped at max_slices.
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask.shape}")
    areas = [(i, int(np.count_nonzero(mask[:, :, i]))) for i in range(mask.shape[2])]
    areas = [x for x in areas if x[1] > 0]
    if not areas:
        return []
    # Sort by area desc and pick top max_slices uniformly distributed
    areas.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in areas[:max_slices * 3]]  # collect more, then downsample to spread coverage
    if len(top_indices) <= max_slices:
        return top_indices
    # Evenly sample across top candidates
    sampled = np.linspace(0, len(top_indices) - 1, num=max_slices, dtype=int)
    return [top_indices[i] for i in sampled]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def overlay_and_save(
    image_vol: np.ndarray,
    mask_vol: np.ndarray,
    slice_indices: Iterable[int],
    out_dir: str,
    subject_id: str,
    alpha: float = 0.4,
    outline: bool = True,
) -> List[Tuple[int, str, str]]:
    """
    Save per-slice base and overlay PNGs.
    Returns list of tuples: (slice_index, base_png_path, overlay_png_path).
    """
    saved: List[Tuple[int, str, str]] = []
    subject_out_dir = os.path.join(out_dir, f"ISPY1_{subject_id}")
    ensure_dir(subject_out_dir)
    img_w = window_image(image_vol)
    for idx in slice_indices:
        if idx < 0 or idx >= img_w.shape[2]:
            continue
        img_slice = img_w[:, :, idx]
        msk_slice = (mask_vol[:, :, idx] > 0).astype(np.uint8)

        if np.count_nonzero(msk_slice) == 0:
            # Skip empty slices to reduce clutter
            continue

        # Save base image
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.imshow(img_slice, cmap="gray", interpolation="nearest")
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        base_path = os.path.join(subject_out_dir, f"ISPY1_{subject_id}_slice_{idx:03d}_base.png")
        fig.savefig(base_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # Save overlay
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.imshow(img_slice, cmap="gray", interpolation="nearest")
        if outline:
            # Draw mask contour lines
            ax.contour(msk_slice, levels=[0.5], colors=["red"], linewidths=1.0)
        else:
            # Semi-transparent filled overlay
            ax.imshow(np.ma.masked_where(msk_slice == 0, msk_slice), cmap="autumn", alpha=alpha, interpolation="nearest")
        ax.set_axis_off()
        fig.tight_layout(pad=0)

        overlay_path = os.path.join(subject_out_dir, f"ISPY1_{subject_id}_slice_{idx:03d}_overlay.png")
        fig.savefig(overlay_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        saved.append((idx, base_path, overlay_path))
    return saved


def create_html_viewer(
    subject_id: str,
    frames: List[Tuple[int, str, str]],
    out_dir: str,
    embed: bool = True,
    show_overlay_by_default: bool = True,
) -> str:
    """
    Create a simple HTML viewer with a slider to browse slices.
    Allows toggling between base and overlay images.
    If embed=True, images are embedded as data URIs; otherwise paths are used.
    Returns path to the created HTML file.
    """
    subject_out_dir = os.path.join(out_dir, f"ISPY1_{subject_id}")
    ensure_dir(subject_out_dir)
    html_path = os.path.join(subject_out_dir, f"ISPY1_{subject_id}_viewer.html")

    # Sort frames by slice index
    frames_sorted = sorted(frames, key=lambda x: x[0])

    def to_data_uri(p: str) -> str:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    base_sources = []
    overlay_sources = []
    for _, base_p, overlay_p in frames_sorted:
        base_sources.append(to_data_uri(base_p) if embed else os.path.basename(base_p))
        overlay_sources.append(to_data_uri(overlay_p) if embed else os.path.basename(overlay_p))

    # Build HTML
    num = len(frames_sorted)
    initial_index = 0
    initial_base = base_sources[initial_index] if num > 0 else ""
    initial_overlay = overlay_sources[initial_index] if num > 0 else ""
    initial_src = initial_overlay if show_overlay_by_default else initial_base

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ISPY1 {subject_id} Viewer</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 16px; background: #111; color: #eee; }}
    .container {{ max-width: 720px; margin: 0 auto; }}
    .controls {{ display: flex; align-items: center; gap: 12px; margin: 12px 0; flex-wrap: wrap; }}
    .img-wrap {{ text-align: center; }}
    img {{ width: 100%; height: auto; image-rendering: auto; background: #000; }}
    input[type="range"] {{ width: 100%; }}
    .toggle {{ display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }}
    .meta {{ opacity: 0.8; font-size: 14px; }}
  </style>
</head>
<body>
  <div class="container">
    <h2>ISPY1 {subject_id} — Slice Viewer</h2>
    <div class="controls">
      <label class="toggle"><input type="checkbox" id="overlayToggle" {"checked" if show_overlay_by_default else ""}> Show overlay</label>
      <span class="meta">Slice: <span id="sliceLabel">{frames_sorted[initial_index][0] if num > 0 else "-"}</span> ({num} frames)</span>
    </div>
    <input id="slider" type="range" min="0" max="{max(0, num-1)}" step="1" value="{initial_index}"/>
    <div class="img-wrap">
      <img id="viewer" src="{initial_src}" alt="slice"/>
    </div>
  </div>
  <script>
    const baseSources = {base_sources};
    const overlaySources = {overlay_sources};
    const frames = { [idx for idx, _, _ in frames_sorted] };
    const slider = document.getElementById('slider');
    const viewer = document.getElementById('viewer');
    const overlayToggle = document.getElementById('overlayToggle');
    const sliceLabel = document.getElementById('sliceLabel');

    function update() {{
      const i = parseInt(slider.value, 10) || 0;
      sliceLabel.textContent = frames[i] ?? '-';
      const useOverlay = overlayToggle.checked;
      const src = useOverlay ? overlaySources[i] : baseSources[i];
      viewer.src = src;
    }}
    slider.addEventListener('input', update);
    overlayToggle.addEventListener('change', update);
  </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def gather_slice_indices(mask_vol: np.ndarray, save_all: bool, num_slices: int) -> List[int]:
    if save_all:
        non_empty = [i for i in range(mask_vol.shape[2]) if np.count_nonzero(mask_vol[:, :, i]) > 0]
        return non_empty
    return find_mask_slices(mask_vol, max_slices=num_slices)


def build_argparser() -> argparse.ArgumentParser:
    repo_root = find_repo_root()
    default_images = os.path.join(repo_root, "ispy1", "images_bias-corrected_nifti")
    default_masks = os.path.join(repo_root, "ispy1", "stv_manual_masks")
    default_out = os.path.join(repo_root, "ispy1", "overlays")

    parser = argparse.ArgumentParser(description="Overlay ISPY1 masks onto bias-corrected NIfTI images and export PNGs and optional HTML viewer.")
    parser.add_argument("--images-root", type=str, default=default_images, help="Path to images root (per-subject folders).")
    parser.add_argument("--masks-root", type=str, default=default_masks, help="Path to masks folder (subject-level .nii.gz).")
    parser.add_argument("--out-dir", type=str, default=default_out, help="Directory to write outputs (PNGs/HTML).")
    parser.add_argument("--preferred-tps", type=str, nargs="*", default=["0002", "0001", "0000"], help="Preferred DCE timepoints in order.")
    parser.add_argument("--subjects", type=str, nargs="*", default=None, help="Subset of subject IDs to process (e.g., 1147 1130).")
    parser.add_argument("--num-slices", type=int, default=6, help="Number of representative slices per subject (ignored with --all).")
    parser.add_argument("--all", dest="save_all", action="store_true", help="Save all slices that contain mask.")
    parser.add_argument("--alpha", type=float, default=0.4, help="Overlay alpha when not using outline.")
    parser.add_argument("--outline", action="store_true", help="Draw mask contours instead of filled overlay.")
    parser.add_argument("--viewer", action="store_true", help="Also generate an interactive HTML slice viewer per subject.")
    parser.add_argument("--no-embed", dest="embed", action="store_false", help="Do not embed images in HTML; reference files instead.")
    parser.add_argument("--qa", action="store_true", help="Compute and save resampling QA metrics (CSV).")
    parser.add_argument("--reorient-canonical", action="store_true", help="Reorient images to RAS+ canonical before resampling.")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    images_root = args.images_root
    masks_root = args.masks_root
    out_dir = args.out_dir
    preferred_tps = args.preferred_tps
    subject_filter = set(args.subjects) if args.subjects else None

    ensure_dir(out_dir)

    subjects = list_subject_ids(images_root)
    if subject_filter:
        subjects = [sid for sid in subjects if sid in subject_filter]

    processed = 0
    skipped = 0
    # Prepare QA CSV if needed
    qa_csv_path = os.path.join(out_dir, "resampling_qc.csv") if getattr(args, "qa", False) else None
    if qa_csv_path and not os.path.exists(qa_csv_path):
        with open(qa_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "subject_id",
                "img_shape", "img_zooms_mm", "img_axcodes",
                "mask_shape", "mask_zooms_mm", "mask_axcodes",
                "mask_rs_shape", "mask_rs_zooms_mm",
                "mask_volume_ml_orig", "mask_volume_ml_rs", "mask_volume_pct_diff",
                "roundtrip_dice",
            ])

    def fmt_tuple(t):
        return "x".join(str(int(x)) for x in t) if hasattr(t, "__iter__") else str(t)

    def get_zooms(img):
        try:
            return tuple(float(z) for z in img.header.get_zooms()[:3])
        except Exception:
            return (np.nan, np.nan, np.nan)

    def voxel_volume_ml_from_img(img):
        # mm^3 per voxel = |det(affine[:3,:3])|; convert to mL (1 mL = 1000 mm^3)
        try:
            mm3 = abs(float(np.linalg.det(img.affine[:3, :3])))
            return mm3 / 1000.0
        except Exception:
            z = get_zooms(img)
            if all(np.isfinite(z)):
                return (z[0] * z[1] * z[2]) / 1000.0
            return float("nan")

    def dice_coef(a: np.ndarray, b: np.ndarray) -> float:
        a = (a > 0).astype(np.uint8)
        b = (b > 0).astype(np.uint8)
        inter = int((a & b).sum())
        size_sum = int(a.sum() + b.sum())
        return (2.0 * inter / size_sum) if size_sum > 0 else 1.0

    for sid in subjects:
        subject_dir = os.path.join(images_root, f"ISPY1_{sid}")
        mask_path = os.path.join(masks_root, f"ISPY1_{sid}.nii.gz")
        if not os.path.isfile(mask_path):
            skipped += 1
            continue
        vol_path = choose_timepoint_file(subject_dir, preferred_tps)
        if vol_path is None or not os.path.isfile(vol_path):
            skipped += 1
            continue
        try:
            image_img_raw = load_nifti_img(vol_path)
            mask_img_raw = load_nifti_img(mask_path)

            image_img = nib.as_closest_canonical(image_img_raw) if args.reorient_canonical else image_img_raw
            mask_img = nib.as_closest_canonical(mask_img_raw) if args.reorient_canonical else mask_img_raw

            # Always resample mask to image grid to handle size/orientation differences
            mask_img_rs = resample_mask_to_image(mask_img, image_img)

            image_vol = image_img.get_fdata(dtype=np.float32)
            mask_vol = mask_img_rs.get_fdata(dtype=np.float32)

            # Binarize mask after resampling (in case of interpolation artifacts)
            mask_vol = (mask_vol > 0.5).astype(np.uint8)
        except Exception as e:
            print(f"[ERROR] {sid}: failed to load NIfTI ({e})")
            skipped += 1
            continue

        # Optional QA: compute volumes and roundtrip Dice
        if qa_csv_path:
            try:
                # Volume in mL (sum(mask) * voxel_volume)
                vox_ml_orig = voxel_volume_ml_from_img(mask_img)
                vox_ml_rs = voxel_volume_ml_from_img(mask_img_rs)
                vol_ml_orig = float(mask_img.get_fdata(dtype=np.float32).sum()) * vox_ml_orig
                vol_ml_rs = float(mask_vol.sum()) * vox_ml_rs
                vol_pct_diff = (100.0 * (vol_ml_rs - vol_ml_orig) / vol_ml_orig) if vol_ml_orig > 0 else float("nan")

                # Roundtrip Dice: resample resampled mask back to original mask space
                mask_back_img = resample_from_to(mask_img_rs, mask_img, order=0)
                mask_back = (mask_back_img.get_fdata(dtype=np.float32) > 0.5).astype(np.uint8)
                mask_orig_bin = (mask_img.get_fdata(dtype=np.float32) > 0.5).astype(np.uint8)
                rt_dice = dice_coef(mask_orig_bin, mask_back)

                with open(qa_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        sid,
                        fmt_tuple(image_vol.shape), fmt_tuple(get_zooms(image_img)), "".join(aff2axcodes(image_img.affine)),
                        fmt_tuple(mask_img.shape), fmt_tuple(get_zooms(mask_img)), "".join(aff2axcodes(mask_img.affine)),
                        fmt_tuple(mask_vol.shape), fmt_tuple(get_zooms(mask_img_rs)),
                        f"{vol_ml_orig:.3f}", f"{vol_ml_rs:.3f}", f"{vol_pct_diff:.2f}",
                        f"{rt_dice:.4f}",
                    ])
            except Exception as e:
                print(f"[WARN] {sid}: QA computation failed ({e})")

        slice_indices = gather_slice_indices(mask_vol, save_all=args.save_all, num_slices=args.num_slices)
        if not slice_indices:
            print(f"[INFO] {sid}: no mask-positive slices — skipping")
            skipped += 1
            continue
        frames = overlay_and_save(
            image_vol=image_vol,
            mask_vol=mask_vol,
            slice_indices=slice_indices,
            out_dir=out_dir,
            subject_id=sid,
            alpha=args.alpha,
            outline=bool(args.outline),
        )
        print(f"[OK] {sid}: saved {len(frames)} base/overlay pairs")
        if args.viewer and frames:
            html_path = create_html_viewer(
                subject_id=sid,
                frames=frames,
                out_dir=out_dir,
                embed=bool(getattr(args, "embed", True)),
                show_overlay_by_default=True,
            )
            print(f"[OK] {sid}: viewer -> {html_path}")
        processed += 1

    print(f"Done. Subjects processed: {processed}, skipped: {skipped}. Output: {out_dir}")


if __name__ == "__main__":
    main()

