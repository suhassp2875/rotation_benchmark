# Model Checkpoints
This directory contains verified pre-trained weights for the `RotationBench` benchmark.

## Verified Models
- `sixd_checkpoint.pt`: 6D Gram-Schmidt representation. Best for general-purpose accuracy and occlusion robustness.
- `lie_fullfix_checkpoint.pt`: Axis-angle representation with gated antipodal loss. Best for worst-case boundary stability near 180°.

## Provenance
Each model's training configuration, date, and commit hash are recorded in `manifest.json`.
