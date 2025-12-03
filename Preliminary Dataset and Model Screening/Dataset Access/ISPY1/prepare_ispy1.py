#!/usr/bin/env python3
"""
Prepare ISPY1 MRI dataset for segmentation:
 - Uses tcia_utils (NBIA) to list and download DICOM series from the ISPY1 collection
 - Converts DICOM to NIfTI with dcm2niix
 - Optionally integrates STV segmentation masks from a local analysis result (NIfTI) path
 - Reorients images and labels to RAS and aligns labels to a chosen reference image grid
 - Writes patient-level train/val/test splits; supports limiting number of patients for quick tests

Notes:
 - Requires: pip install tcia_utils
 - STV masks (NIfTI) are typically distributed via the ISPY1-Tumor-SEG-Radiomics analysis result.
   Place them under a local directory and pass with --stv-root.
"""
import argparse
import os
import sys
import json
import time
import shutil
import random
import zipfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import zipfile
from tcia_utils import nbia as tcia_nb  # type: ignore

def run_dcm2niix(dicom_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Output name pattern includes protocol (%p) and series number (%s)
    cmd = ["dcm2niix", "-z", "y", "-f", "%p_%s", "-o", str(out_dir), str(dicom_dir)]
    subprocess.run(cmd, check=True)


def as_ras_inplace(nifti_path: Path) -> None:
    img = nib.load(str(nifti_path))
    ras = nib.as_closest_canonical(img)
    nib.save(ras, str(nifti_path))


def resample_label_to_ref_nn(label_path: Path, ref_path: Path, out_path: Path) -> None:
    ref = sitk.ReadImage(str(ref_path))
    lab = sitk.ReadImage(str(label_path))
    res = sitk.Resample(
        lab,
        ref,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    sitk.WriteImage(res, str(out_path), useCompression=True)


def select_reference_image(nifti_dir: Path) -> Optional[Path]:
    # Prefer sequences commonly used as reference
    preferred_names = ["Pre", "SUB", "Sub_1", "POST_1", "POST", "T2"]
    existing = {p.stem.upper(): p for p in nifti_dir.glob("*.nii.gz")}
    for pref in preferred_names:
        # Match by prefix
        for name, p in existing.items():
            if name.startswith(pref.upper()):
                return p
    # Fallback to first NIfTI
    any_nii = list(nifti_dir.glob("*.nii.gz"))
    return any_nii[0] if any_nii else None


def detect_seq_name(series_desc: Optional[str]) -> Optional[str]:
    if not series_desc:
        return None
    s = series_desc.upper()
    if "PRE" in s:
        return "Pre"
    if "SUB" in s:
        return "Sub_1"
    if "T2" in s:
        return "T2"
    if "POST_2" in s or "PHASE 2" in s or "P2" in s:
        return "Post_2"
    if "POST" in s or "PHASE 1" in s or "P1" in s:
        return "Post_1"
    return None


def find_stv_label_for_patient(stv_root: Path, patient_id: str) -> Optional[Path]:
    """
    Heuristic to find STV label NIfTI for a patient under stv_root.
    Tries common layout patterns, e.g. stv_root/ISPY1_<ID>/lesion.nii.gz or *_seg.nii.gz.
    """
    candidates: List[Path] = []
    pid_norms = {patient_id, patient_id.replace("ISPY1_", ""), f"ISPY1_{patient_id}"}
    # Search subdirs whose name contains patient ID token
    for sub in stv_root.rglob("*"):
        if not sub.is_dir():
            continue
        name_up = sub.name.upper()
        if any(tok.upper() in name_up for tok in pid_norms):
            for nii in sub.glob("*.nii*"):
                n = nii.name.lower()
                if any(k in n for k in ["stv", "lesion", "label", "seg"]):
                    candidates.append(nii)
    if candidates:
        # Prefer names containing "stv"
        for c in candidates:
            if "stv" in c.name.lower():
                return c
        return candidates[0]
    return None


def write_splits(all_patients: List[str], out_dir: Path, val_frac: float, test_frac: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    rnd = random.Random(seed)
    pts = list(all_patients)
    rnd.shuffle(pts)
    n_total = len(pts)
    n_test = int(round(n_total * test_frac))
    n_val = int(round((n_total - n_test) * val_frac))
    test_pts = pts[:n_test]
    val_pts = pts[n_test:n_test + n_val]
    train_pts = pts[n_test + n_val:]
    (out_dir / "lists").mkdir(parents=True, exist_ok=True)
    (out_dir / "lists" / "train.txt").write_text("\n".join(train_pts) + "\n", encoding="utf-8")
    (out_dir / "lists" / "val.txt").write_text("\n".join(val_pts) + "\n", encoding="utf-8")
    (out_dir / "lists" / "test.txt").write_text("\n".join(test_pts) + "\n", encoding="utf-8")
    return train_pts, val_pts, test_pts


def main():
    parser = argparse.ArgumentParser(description="Download and prepare ISPY1 MRI with STV labels for segmentation.")
    parser.add_argument("--collection", default="ISPY1", help="TCIA collection name.")
    parser.add_argument("--out-root", required=True, help="Output root directory.")
    parser.add_argument("--stv-root", default=None, help="Path to local STV NIfTI analysis result (optional).")
    parser.add_argument("--max-patients", type=int, default=None, help="Limit total patients for quick tests.")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction (post-test split).")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Test fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits.")
    parser.add_argument("--only-list", action="store_true", help="Only list patients and exit.")
    parser.add_argument("--patients-list", default=None, help="Path to a text file with PatientIDs to use (bypass API).")
    parser.add_argument("--skip-download", action="store_true", help="Skip DICOM downloads (assume present).")
    parser.add_argument("--skip-convert", action="store_true", help="Skip DICOM->NIfTI conversion (assume present).")
    parser.add_argument("--skip-stv", action="store_true", help="Skip STV label integration.")
    args = parser.parse_args()

    # Require tcia_utils
    try:
        from tcia_utils import nbia as tcia_nb  # type: ignore
    except Exception as e:
        print(f"[ERROR] tcia_utils is required. Install with: pip install tcia_utils. Detail: {e}", file=sys.stderr)
        sys.exit(1)
    out_root = Path(args.out_root)
    dicom_root = out_root / "raw_dicom"
    nifti_root = out_root / "nifti"
    labels_root = out_root / "labels"
    logs_root = out_root / "logs"
    for p in (dicom_root, nifti_root, labels_root, logs_root):
        p.mkdir(parents=True, exist_ok=True)

    # Helpers to find columns in dataframes from tcia_utils
    def _is_dataframe(obj) -> bool:
        try:
            import pandas as _pd  # local import to avoid hard dep
            return isinstance(obj, _pd.DataFrame)
        except Exception:
            return False

    def _df_find_col(df, preferred: Tuple[str, ...], fuzzy_contains: Tuple[str, ...] = ()) -> Optional[str]:
        try:
            cols = list(getattr(df, "columns", []))
        except Exception:
            return None
        lower_map = {c.lower(): c for c in cols}
        for cand in preferred:
            c = lower_map.get(cand.lower())
            if c:
                return c
        tokens = [t.lower() for t in fuzzy_contains]
        if tokens:
            for c in cols:
                cl = c.lower()
                if all(t in cl for t in tokens):
                    return c
        return None

    def _extract_patient_ids(obj) -> List[str]:
        """
        Accepts either a pandas DataFrame or a list of dicts and returns sorted PatientIDs.
        """
        # DataFrame
        if _is_dataframe(obj):
            pcol = _df_find_col(
                obj,
                preferred=("PatientID", "patientId", "patientID", "patientid", "Patient Id"),
                fuzzy_contains=("patient", "id"),
            )
            if not pcol:
                raise RuntimeError("Could not find PatientID column in tcia_utils result.")
            return sorted(set(obj[pcol].astype(str).tolist()))
        # List of dicts
        if isinstance(obj, list):
            ids = set()
            for row in obj:
                if not isinstance(row, dict):
                    continue
                pid = (
                    row.get("PatientID")
                    or row.get("patientId")
                    or row.get("patientID")
                    or row.get("Patient Id")
                    or row.get("SubjectID")
                    or row.get("subjectId")
                )
                if pid:
                    ids.add(str(pid))
            if not ids:
                raise RuntimeError("No PatientID keys found in tcia_utils list result.")
            return sorted(ids)
        raise RuntimeError(f"Unsupported object type for patient listing: {type(obj)}")

    def _extract_series_list(obj) -> List[Dict[str, Optional[str]]]:
        """
        Accepts either a pandas DataFrame or a list of dicts and returns a normalized list of series dicts:
        {seriesInstanceUid, seriesDescription, modality}
        """
        results: List[Dict[str, Optional[str]]] = []
        if _is_dataframe(obj):
            uid_col = _df_find_col(
                obj,
                preferred=("SeriesInstanceUID", "seriesInstanceUid", "seriesInstanceUID", "seriesinstanceuid"),
                fuzzy_contains=("series", "uid"),
            )
            desc_col = _df_find_col(
                obj, preferred=("SeriesDescription", "seriesDescription", "seriesdescription"), fuzzy_contains=("description",)
            )
            mod_col = _df_find_col(obj, preferred=("Modality", "modality"))
            for _, row in obj.iterrows():
                uid = str(row[uid_col]) if uid_col else None
                if not uid:
                    continue
                entry = {
                    "seriesInstanceUid": uid,
                    "seriesDescription": str(row[desc_col]) if desc_col else None,
                    "modality": (str(row[mod_col]) if mod_col else None) or None,
                }
                results.append(entry)
            return results
        if isinstance(obj, list):
            for row in obj:
                if not isinstance(row, dict):
                    continue
                uid = (
                    row.get("SeriesInstanceUID")
                    or row.get("seriesInstanceUid")
                    or row.get("seriesInstanceUID")
                    or row.get("seriesinstanceuid")
                )
                if not uid:
                    continue
                desc = row.get("SeriesDescription") or row.get("seriesDescription") or row.get("seriesdescription")
                mod = row.get("Modality") or row.get("modality")
                results.append(
                    {
                        "seriesInstanceUid": str(uid),
                        "seriesDescription": str(desc) if desc is not None else None,
                        "modality": str(mod) if mod is not None else None,
                    }
                )
            return results
        raise RuntimeError(f"Unsupported object type for series listing: {type(obj)}")

    if args.patients_list and Path(args.patients_list).exists():
        print(f"[INFO] Loading patients from file: {args.patients_list}")
        patient_ids = [ln.strip() for ln in Path(args.patients_list).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        print(f"[INFO] Listing patients in collection '{args.collection}'...")
        try:
            # Prefer dedicated patient listing if available
            dfp = None
            for kw in ({"collection": args.collection}, {"Collection": args.collection}):
                try:
                    dfp = tcia_nb.getPatient(**kw)
                    break
                except Exception:
                    dfp = None
            if dfp is not None:
                patient_ids = _extract_patient_ids(dfp)
            else:
                # Fallback to series listing and derive unique patients
                dfs = None
                for kw in ({"collection": args.collection}, {"Collection": args.collection}):
                    try:
                        dfs = tcia_nb.getSeries(**kw)
                        break
                    except Exception:
                        dfs = None
                if dfs is None:
                    raise RuntimeError("tcia_utils.getSeries failed for collection listing.")
                # Extract patient IDs from the series listing object
                series_norm = _extract_series_list(dfs)
                # derive patient ids from original object if available; otherwise fall back to none
                ids = set()
                if _is_dataframe(dfs):
                    ids = set(_extract_patient_ids(dfs))
                elif isinstance(dfs, list):
                    ids = set(_extract_patient_ids(dfs))
                else:
                    raise RuntimeError("Unsupported getSeries return type.")
                patient_ids = sorted(ids)
        except Exception as e:
            print(f"[ERROR] tcia_utils failed to list patients: {e}", file=sys.stderr)
            sys.exit(1)
    patient_ids.sort()
    print(f"[INFO] Found {len(patient_ids)} patients.")

    if args.max_patients is not None:
        patient_ids = patient_ids[: args.max_patients]
        print(f"[INFO] Limiting to first {len(patient_ids)} patients for this run.")

    if args.only_list:
        print("\n".join(patient_ids))
        return

    stv_root = Path(args.stv_root) if args.stv_root else None
    processed_patients: List[str] = []

    for pid in patient_ids:
        print(f"[INFO] Processing patient: {pid}")
        patient_dicom_dir = dicom_root / pid
        patient_nifti_dir = nifti_root / pid
        patient_label_dir = labels_root / pid
        patient_dicom_dir.mkdir(parents=True, exist_ok=True)
        patient_nifti_dir.mkdir(parents=True, exist_ok=True)
        patient_label_dir.mkdir(parents=True, exist_ok=True)

        # Enumerate MR series via tcia_utils
        try:
            df = None
            for kw in (
                {"collection": args.collection, "PatientID": pid},
                {"collection": args.collection, "patientId": pid},
                {"Collection": args.collection, "PatientID": pid},
            ):
                try:
                    df = tcia_nb.getSeries(**kw)
                    break
                except TypeError:
                    continue
            if df is None:
                raise RuntimeError("tcia_utils.getSeries failed for patient.")
            series_norm = _extract_series_list(df)
            # filter MR
            mr_series = [s for s in series_norm if (s.get("modality") or "").upper() == "MR"]
        except Exception as e:
            print(f"[WARN] tcia_utils listing failed for {pid}: {e}")
            mr_series = []
        if not mr_series:
            print(f"[WARN] No MR series for {pid}, skipping.")
            continue

        # Download each series using tcia_utils
        if not args.skip_download:
            for s in mr_series:
                uid = s["seriesInstanceUid"]
                sdir = patient_dicom_dir / uid
                try:
                    sdir.mkdir(parents=True, exist_ok=True)
                    downloaded = False
                    try:
                        tcia_nb.downloadSeries(seriesInstanceUid=uid, path=str(sdir))
                        downloaded = True
                    except TypeError:
                        try:
                            tcia_nb.downloadSeries(uid, str(sdir))
                            downloaded = True
                        except Exception:
                            pass
                    if not downloaded:
                        raise RuntimeError("tcia_utils.downloadSeries did not accept parameters used.")
                except Exception as e:
                    print(f"[WARN]  tcia_utils download failed for {uid}: {e}")

        # Convert with dcm2niix
        if not args.skip_convert:
            for s in mr_series:
                uid = s["seriesInstanceUid"]
                sdir = patient_dicom_dir / uid
                if not sdir.exists():
                    continue
                try:
                    run_dcm2niix(sdir, patient_nifti_dir)
                except subprocess.CalledProcessError as e:
                    print(f"[WARN]  dcm2niix failed for {uid}: {e}")

        # Reorient produced NIfTIs to RAS
        for nii in patient_nifti_dir.glob("*.nii.gz"):
            try:
                as_ras_inplace(nii)
            except Exception as e:
                print(f"[WARN]  RAS reorient failed for {nii.name}: {e}")

        # Integrate STV label if provided
        if not args.skip_stv and stv_root and stv_root.exists():
            label_src = find_stv_label_for_patient(stv_root, pid)
            if label_src and label_src.exists():
                ref_img = select_reference_image(patient_nifti_dir)
                if ref_img is None:
                    # No reference found; copy raw label
                    dst = patient_label_dir / "lesion.nii.gz"
                    try:
                        shutil.copy2(label_src, dst)
                    except Exception as e:
                        print(f"[WARN]  Failed to copy label for {pid}: {e}")
                else:
                    # Ensure label is RAS then resample to ref
                    tmp_ras = patient_label_dir / "_tmp_label_ras.nii.gz"
                    try:
                        # Save RAS-converted label to tmp path
                        lbl_img = nib.load(str(label_src))
                        lbl_ras = nib.as_closest_canonical(lbl_img)
                        nib.save(lbl_ras, str(tmp_ras))
                        out_lbl = patient_label_dir / "lesion.nii.gz"
                        resample_label_to_ref_nn(tmp_ras, ref_img, out_lbl)
                        tmp_ras.unlink(missing_ok=True)
                    except Exception as e:
                        print(f"[WARN]  Label resample failed for {pid}: {e}")
            else:
                print(f"[INFO]  No STV label found for {pid} under {stv_root}")

        processed_patients.append(pid)

    # Write splits
    if processed_patients:
        train_pts, val_pts, test_pts = write_splits(processed_patients, out_root, args.val_frac, args.test_frac, args.seed)
        meta = {
            "collection": args.collection,
            "num_patients_total": len(processed_patients),
            "splits": {"train": len(train_pts), "val": len(val_pts), "test": len(test_pts)},
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (out_root / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[INFO] Split written. Train={len(train_pts)} Val={len(val_pts)} Test={len(test_pts)}")
    else:
        print("[WARN] No patients processed; nothing to split.")


if __name__ == "__main__":
    main()


