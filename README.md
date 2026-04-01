# RotationBench: Reliability Benchmark for 3D Rotation Representations

[![CoRL 2026](https://img.shields.io/badge/CoRL-2026-blue.svg)](https://corl2026.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RotationBench** is a diagnostic framework designed to expose the operational vulnerabilities of neural rotation representations. While standard benchmarks often prioritize median accuracy, they frequently mask catastrophic failures in edge-case orientations. Our framework moves beyond aggregate metrics to characterize **benchmark tail-risk ($p_{99}$)**, providing a rigorous instrument to forecast robotic failures before deployment.

By employing a massive 25-million sample SO(3) stratified sweep, the benchmark isolates topological discontinuities and gimbal singularities that cause minimal representations (e.g., Euler angles or Quaternions) to collapse. We demonstrate that this representational tail-risk directly predicts the "Success Cliff" in downstream robotic task success for `PickPlace` and `Stack`.

---

## 🚀 Quick Start

Evaluate any SO(3) model checkpoint against the stratified benchmark:

```python
from rotation_benchmark.eval.benchmark import evaluate_model

# Run the 100k-sample stratified benchmark
# Returns median, p99, and failure rate (>30°)
evaluate_model(
    model_path="checkpoints/sixd_checkpoint.pt",
    rep_type="6d",
    num_samples=100000,
    bootstrap=True
)
```

For full reproduction of the manuscript figures, use the provided scripts:
```bash
python scripts/plot_benchmark_summary.py   # Figure 1: Summary
python scripts/visualize_discontinuity.py # Figure 2: Mechanics
python scripts/plot_error_cliff.py        # Figure 3: Hero Cliff
python scripts/plot_p99_vs_success.py     # Figure 4: Correlation
```

---

## 📊 The RISK Table

The following table summarizes the operational reliability of standard representations based on their benchmark $p_{99}$ tail-risk relative to downstream task "success cliffs."

| Representation | Benchmark $p_{99}$ (Uniform) | PickPlace (21.1° Cliff) | Stack (30.4° Cliff) |
| :--- | :--- | :--- | :--- |
| **6D / SVD** | ~4° | **SAFE** ✅ | **SAFE** ✅ |
| **Atlas** | 30.6° | **RISK** ⚠️ | **RISK (Marginal)** ⚠️ |
| **Lie (fixed)** | 31.9° | **RISK** ⚠️ | **RISK** ⚠️ |
| **Quaternion** | 46.1° | **RISK** ⚠️ | **RISK** ⚠️ |
| **Euler** | 118.8° | **CRITICAL** ❌ | **CRITICAL** ❌ |

---

## 🛠️ Core Components

1.  **Stratified Benchmark**: Synthetic SO(3) evaluation across *Safe*, *Gimbal*, and *Antipodal* regions to measure representational stability.
2.  **Mechanistic Probes**: Single-axis boundary sweeps to isolate topological discontinuities and singularity-driven spikes.
3.  **Success Cliff Analysis**: High-resolution downstream evaluation using **Robosuite** (MuJoCo) to map orientation error to task failure.
4.  **Predictive Correlation**: Statistical validation ($\rho = -0.952$) linking benchmark metrics to operational robot success.

---

## 📦 Pre-Trained Checkpoints

We provide the optimized weights used in our CoRL 2026 manuscript:
*   `checkpoints/sixd_checkpoint.pt`: Continuous 6D representation.
*   `checkpoints/svd_checkpoint.pt`: SVD-based orthogonalization.
*   `checkpoints/atlas_checkpoint.pt`: Multi-chart Atlas mapping.
*   `checkpoints/lie_fullfix_checkpoint.pt`: Robust Axis-Angle with gated loss.
*   `outputs/euler/best.pt`: Standard 3-Euler angle mapping.

---

## 📂 Repository Structure

*   `src/rotation_benchmark/`: Core geometry kernels, SO(3) metrics, and model factory.
*   `scripts/`: Publication-ready figure generation and reproduction tools.
*   `results/final_manuscript_figures/`: High-resolution figures and final manuscript draft.
*   `configs/`: Hyperparameter configurations for SO(3) training.

---

## 📝 Citation

If you use this benchmark or the provided checkpoints in your research, please cite:

```bibtex
@inproceedings{sp2026rotationbench,
  title={Operational Reliability of 3D Rotation Representations in Robotics},
  author={Suhas S P and others},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2026}
}
```
