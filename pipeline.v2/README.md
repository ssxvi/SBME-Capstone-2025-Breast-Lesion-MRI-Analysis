# MRI Analysis Pipeline

Automated breast MRI lesion detection, segmentation, and malignancy classification.

```
Stage 1 → Preprocessing (preprocess.py)
Stage 2 → Lesion detection — EfficientNet-B0 binary (classify_lesion.py)
Stage 3 → Segmentation    — nnUNet (segment.py)          [if lesion found]
Stage 4 → Malignancy      — EfficientNet-B0 binary (classify_malignancy.py) [if lesion found]
Stage 5 → HTML report     (report.py + templates/report.html.j2)
```

---

## Project layout

```
project/
├── pipeline/
│   ├── preprocess.py           # MIP computation (refactored from newprecomputemips.py)
│   ├── classify_lesion.py      # Stage 2: lesion vs no-lesion EfficientNet
│   ├── segment.py              # Stage 3: nnUNet inference via CLI subprocess
│   ├── classify_malignancy.py  # Stage 4: malignant vs benign EfficientNet
│   ├── report.py               # Stage 5: Jinja2 HTML report generator
│   └── run_pipeline.py         # Orchestrator — chains all stages
├── api/
│   ├── main.py                 # FastAPI app (upload / run / result / report endpoints)
│   └── schemas.py              # Pydantic request & response models
├── templates/
│   └── report.html.j2          # HTML report template
├── weights/                    # Put your .pth model weights here
│   ├── lesion_classifier.pth
│   └── malignancy_classifier.pth
├── uploads/                    # Auto-created — uploaded .nii.gz files land here
├── results/                    # Auto-created — one subdirectory per run
└── requirements.txt
```

---

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

Interactive docs: http://localhost:8000/docs

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
    })
}).then(r => r.json());
```

### 3. Poll for results

```js
let result;
while (true) {
    result = await fetch(`/result/${run_id}`).then(r => r.json());
    if (["complete", "failed"].includes(result.status)) break;
    await new Promise(r => setTimeout(r, 2000));  // poll every 2 s
}
```

### 4. Fetch the HTML report

```js
// Open in a new tab:
window.open(`/report/${run_id}`, "_blank");

// Or embed in an <iframe>:
// <iframe src={`/report/${run_id}`} />
```

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
| `SLURM_CPUS_PER_TASK` | `4` | Workers for batch MIP precomputation |
