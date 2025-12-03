ISPY1 preparation (TCIA REST) for segmentation
================================================

This directory contains a script to:
- list and download ISPY1 DICOM via TCIA REST,
- convert to NIfTI (`dcm2niix`),
- optionally merge STV segmentation masks (NIfTI) from the ISPY1-Tumor-SEG-Radiomics analysis result,
- reorient to RAS and align labels to a chosen reference image grid,
- and write patient-level train/val/test splits.

References
- ISPY1 collection (DICOM, ~78 GB): `https://www.cancerimagingarchive.net/collection/ispy1/`
- ISPY1-Tumor-SEG-Radiomics (NIfTI STV labels, ~6.1 GB): `https://www.cancerimagingarchive.net/analysis-result/ispy1-tumor-seg-radiomics/`

Prerequisites
- TCIA API Key is optional for public datasets (you can omit it).
- `dcm2niix` available on PATH (module or conda).
- Python packages: `requests`, `nibabel`, `SimpleITK`.

Environment suggestion (Sockeye)
```bash
mamba create -n ispy1 python=3.10 requests nibabel SimpleITK -c conda-forge -y
# dcm2niix via conda if module not available:
mamba install -n ispy1 -c conda-forge dcm2niix -y
```

One-time (login/DTN) – Download and prepare (limit to N patients)
```bash
OUT=/project/rrg-<pi>/<user>/ispy1
STV=/project/rrg-<pi>/<user>/ispy1_seg_nifti   # unpack of analysis result; optional

python "Preliminary Dataset and Model Screening/Dataset Access/ISPY1/prepare_ispy1.py" \
  --out-root "$OUT" \
  --stv-root "$STV" \
  --max-patients 10 \
  --val-frac 0.1 --test-frac 0.1
```

Notes
- If compute nodes have no internet, run the download step on login/DTN. You can later use `--skip-download` in batch jobs.
- STV masks are discovered heuristically under `--stv-root`. Adjust folder names if needed.

Sockeye batch (conversion/repack/splits without downloading)
- Use `ispy1_prepare.sbatch` to run conversion/reorientation/resampling and splits on compute nodes (no internet).
```bash
sbatch "Preliminary Dataset and Model Screening/Dataset Access/ISPY1/ispy1_prepare.sbatch"
```

Quick tests
- List patients only:
```bash
python "Preliminary Dataset and Model Screening/Dataset Access/ISPY1/prepare_ispy1.py" \
  --only-list --out-root ./out
```
- Run on 5 patients, no labels:
```bash
python "Preliminary Dataset and Model Screening/Dataset Access/ISPY1/prepare_ispy1.py" \
  --out-root ./out --max-patients 5 --val-frac 0.1 --test-frac 0.1
```

Outputs
- DICOM: `out-root/raw_dicom/<PatientID>/<SeriesUID>/...`
- NIfTI (RAS): `out-root/nifti/<PatientID>/*.nii.gz`
- Labels (aligned): `out-root/labels/<PatientID>/lesion.nii.gz`
- Splits: `out-root/lists/{train,val,test}.txt`
- Metadata: `out-root/dataset_meta.json`


