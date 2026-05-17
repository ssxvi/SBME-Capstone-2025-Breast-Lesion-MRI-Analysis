# Mock Segmentation Assets

USED ONLY FOR IN-PERSON DEMO AS EXAMPLE

Supported files:
- `segmentation_overlay.png` (or `.jpg` / `.jpeg`): preview image returned in API result.
- `demo_mask.nii.gz` (or `demo_mask.nii`): optional 3D binary/label mask used to compute area and volume.
- `demo_mask.npy`: optional legacy fallback 3D binary/label mask array `(H, W, D)`.
- `report.html`: optional full HTML report override used for accelerated runs.