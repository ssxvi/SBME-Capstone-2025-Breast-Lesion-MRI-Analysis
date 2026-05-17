# SBME CAPSTONE 2026 - Breast Lesion MRI Analysis Pipeline

## Installation

```bash
pip install -r requirements.txt
```

### nnUNet (separate setup)

nnUNet requires its own environment variables and trained weights.

```bash
pip install nnunetv2

export nnUNet_raw=/data/nnunet/raw
export nnUNet_preprocessed=/data/nnunet/preprocessed
export nnUNet_results=/data/nnunet/results

# Verify:
nnUNetv2_predict --help
```

Place your trained nnUNet model under:
```
$nnUNet_results/Dataset<ID>_<name>/nnUNetTrainer__nnUNetPlans__<config>/fold_<N>/
```

Set the dataset ID and configuration:
```bash
export NNUNET_DATASET_ID=001
export NNUNET_CONFIGURATION=3d_fullres
export NNUNET_FOLD=0
```

### Model weights

Place EfficientNet `.pth` files in `weights/` or point to them via env vars:
```bash
export LESION_MODEL_PATH=weights/lesion_classifier.pth
export MALIGNANCY_MODEL_PATH=weights/malignancy_classifier.pth
```

Weights are loaded as `torch.load(..., map_location="cpu")`. Both raw state dicts and
`{"model_state_dict": ..., "optimizer_state_dict": ...}` checkpoint formats are supported.

---

## Running the API server

```bash
# From the project root:
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API usage (from your React frontend)

### 1. Upload timepoint files

```js
// Call three times — once per timepoint
const form = new FormData();
form.append("file", preFile);   // .nii.gz File object
const { server_path: prePath } = await fetch("/upload", {
    method: "POST", body: form
}).then(r => r.json());
// repeat for post1Path and post2Path
```

### 2. Submit a pipeline run

```js
const { run_id } = await fetch("/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
        case_id:    "patient_001",
        pre_path:   prePath,
        post1_path: post1Path,
        post2_path: post2Path,
        demo_mode: true,          // optional: simulate stages (fast demo)
        demo_duration_sec: 12,    // optional: fake total runtime
    })
}).then(r => r.json());
```

### 3. Poll for results

```js
let result;
while (true) {
    result = await fetch(`/result/${run_id}`).then(r => r.json());
    // result.current_stage -> e.g. "segmentation"
    // result.progress      -> 0.0 ... 1.0
    if (["complete", "failed"].includes(result.status)) break;
    await new Promise(r => setTimeout(r, 2000));  // poll every 2 s
}

// Final result includes segmentation.lesion_area_pixels (2-D projected mask area)
```

### 4. Fetch the HTML report

```js
// Open in a new tab:
window.open(`/report/${run_id}`, "_blank");

// Or embed in an <iframe>:
// <iframe src={`/report/${run_id}`} />
```

### 5. Demo mode notes

- Set `demo_mode: true` in `/run` to simulate progress without running heavy segmentation. Can be changed in the UI directly.
- The API still returns a realistic final payload (lesion, segmentation, malignancy, report).
- Dedicated asset folder (default): `pipeline.v2/mock_segmentation/`
- Supported assets in that folder:
    - `segmentation_overlay.png` (or `.jpg` / `.jpeg`) for UI preview image
    - `demo_mask.npy` for area/volume metrics
    - `report.html` to override demo report HTML
- You can override the folder via env var `PIPELINE_MOCK_SEGMENTATION_DIR`.
- You can also provide a pre-made HTML report via env var `PIPELINE_DEMO_REPORT_HTML`.

---

## CLI usage (single patient, no API)

```bash
python -m pipeline.run_pipeline \
    --case-id patient_001 \
    --pre   /data/raw/pre/patient_001.nii.gz \
    --post1 /data/raw/post1/patient_001.nii.gz \
    --post2 /data/raw/post2/patient_001.nii.gz \
    --output-dir ./results/patient_001
```

Add `--skip-seg` to bypass nnUNet (classification only).

---

## Batch MIP precomputation (training data)

```bash
python -m pipeline.preprocess \
    --raw-root    /data/raw_all \
    --output-root /data/mip_cache \
    --json-root   /data/jsons \
    --workers 8
```

This is equivalent to the original `newprecomputemips.py` behaviour.

---

## nnUNet file naming

The pipeline stages input volumes as:
```
imagesTs/
  <case_id>_0000.nii.gz   ← Pre
  <case_id>_0001.nii.gz   ← Post_1
  <case_id>_0002.nii.gz   ← Post_2
```

Make sure your `dataset.json` `channel_names` matches this order:
```json
{
  "channel_names": { "0": "Pre", "1": "Post_1", "2": "Post_2" }
}
```

---

## Environment variable reference

| Variable | Default | Description |
|---|---|---|
| `LESION_MODEL_PATH` | `weights/lesion_classifier.pth` | Lesion EfficientNet weights |
| `MALIGNANCY_MODEL_PATH` | `weights/malignancy_classifier.pth` | Malignancy EfficientNet weights |
| `nnUNet_raw` | *(required)* | nnUNet raw data root |
| `nnUNet_preprocessed` | *(required)* | nnUNet preprocessed root |
| `nnUNet_results` | *(required)* | nnUNet results/weights root |
| `NNUNET_DATASET_ID` | `001` | nnUNet dataset ID (3 digits) |
| `NNUNET_CONFIGURATION` | `3d_fullres` | nnUNet plan configuration |
| `NNUNET_FOLD` | `0` | Which fold's weights to use |
| `PIPELINE_WORKERS` | `2` | Concurrent pipeline jobs in the API |
| `PIPELINE_MOCK_SEGMENTATION_DIR` | `pipeline.v2/mock_segmentation` | Folder containing accelerated-run segmentation assets |
| `PIPELINE_DEMO_REPORT_HTML` | *(optional)* | Path to a prebuilt HTML report to serve in demo mode |
| `SLURM_CPUS_PER_TASK` | `4` | Workers for batch MIP precomputation |
