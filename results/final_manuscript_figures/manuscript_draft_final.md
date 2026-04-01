# Operational Reliability of 3D Rotation Representations in Robotics (v3.0)

## Abstract
Standard benchmarks for supervised rotation learning emphasize median accuracy, which fails to predict catastrophic operational failures in downstream robotic tasks. We provide a **predictive framework** linking benchmark tail-risk metrics ($p_{99}$) to task-specific orientation error tolerance. Through high-resolve SO(3) perturbation sweeps ($N=100$) across manipulation tasks, we identify "Success Cliff" boundaries and demonstrate that benchmark tail evaluation provides a rigorous instrument for forecasting robotic failures near representational singularities.

## 1. Introduction
Modern robotics increasingly relies on deep learning for 3D pose estimation and control. While 6D and SVD-based representations facilitate continuous optimization, minimal representations remain attractive for their compactness. In this work, we move beyond aggregate accuracy to characterize **worst-case reliability**. We show that a representation's performance in a stratified benchmark directly predicts its success rate in real-world manipulation tasks.

Learned rotation representations have evolved significantly since the identification of the topological limits of minimal parameterizations in high-dimensional deep learning. **Zhou et al. (2019)** proved that any $SO(3)$ representation with fewer than five dimensions must be discontinuous, and introduced the 6D continuous representation. **Levinson et al. (2020)** subsequently proposed an SVD-based orthogonalization method that maintains continuity while ensuring $SO(3)$ validity, showing its effectiveness in object pose estimation. Recent category-level pose estimation architectures, such as **MonoCon (Liu et al. 2022)**, have integrated these continuous representations to improve end-to-end performance. However, these works primarily target aggregate metrics; our work bridges this gap by characterizing the operational tail-risk of these representations in active robotic manipulation.

## 2. The Predictive Benchmark (Measurement Instrument)
We introduce a stratified benchmark that evaluates representations across specific geometric regimes. Our protocol isolates the regions where topological constraints (e.g., gimbal lock, sign ambiguity) manifest as high-error tail events. Using a massive 25-million sample framework, we show that "Safe" region performance is a poor proxy for "Worst-Case" reliability.

## 3. Main Benchmark Summary
Across our large-scale evaluation, we observe a fundamental **Reliability Gap**. Figure 1 illustrates the contrast between safe-region parity and tail divergence. While all methods achieve sub-2.5° median errors in stable regions, minimal mappings exhibit catastrophic tail expansion at representational boundaries (e.g., Euler $p_{99} = 118.8°$, Lie (fixed) $p_{99} = 31.9°$).

![Figure 1: Main Benchmark Summary. A 3-panel comparison shows that while safe-region performance is nearly identical (Panel A), tail-risk (Panel B) and operational failure rates (Panel C) diverge sharply, separating continuous methods from minimal mappings. The 30° success threshold in Panel C corresponds to the characteristic operational cliff for the Stack task.](file:///c:/Users/suhas/Downloads/rotation-benchmark/rotation-benchmark/results/final_manuscript_figures/fig1_benchmark_summary.png)

## 4. Failure Anatomy: Mechanistic Analysis
The tail-risk divergence is driven by **output-space discontinuities**. Figure 2 provides mechanistic proof of this collapse near the 180° antipodal boundary. Using a single-axis probe sweep, we isolate the topological failure: while continuous mappings (6D) remain stable, minimal representations (Euler, Lie (fixed)) exhibit a catastrophic geodesic error spike exceeding 150°. 

![Figure 2: Mechanistic Discontinuity Proof. Minimal representations exhibit catastrophic geodesic error spikes near representational boundaries (e.g., gimbal singularities and antipodal cut-offs). Note that Quaternion failure is stochastic and axis-dependent, contributing to its unpredictable p99 tail. SVD is omitted for clarity as it exhibits the same flat, low-error behavior as 6D in this probe; Atlas is omitted because it does not exhibit a single isolated boundary mechanism comparable to the topological and singularity failures shown here.](file:///c:/Users/suhas/Downloads/rotation-benchmark/rotation-benchmark/results/final_manuscript_figures/fig2_mechanistic_discontinuity.png)

## 5. Downstream Validation: Success Cliffs
We resolve the **Orientation Success Cliff** for two manipulation tasks: **PickPlace (21.1° cliff)** and **Stack (30.4° cliff)**. Using $N=100$ smoothed trials per bin, we observe a clean monotonic decay in success as orientation error increases. A representation is categorized as **RISK** if its benchmark $p_{99}$ exceeds the task's physical success cliff (Table 1). Notably, **Atlas ($p_{99} = 30.6°$)** sits precisely at the Stack cliff boundary, explainings its marginal reliability and high variance in success compared to continuous methods. This "near-cliff" behavior results in a significant reliability gap that median metrics (where Atlas remains competitive) fail to capture.

| Representation | Benchmark $p_{99}$ (Uniform) | PickPlace (21.1° Cliff) | Stack (30.4° Cliff) |
| :--- | :--- | :--- | :--- |
| **6D / SVD** | ~4° | **SAFE** | **SAFE** |
| **Atlas** | 30.6° | **RISK** | **RISK (Marginal)** |
| **Lie (fixed)** | 31.9° | **RISK** | **RISK** |
| **Quaternion** | 46.1° | **RISK** | **RISK** |
| **Euler** | 118.8° | **CRITICAL** | **CRITICAL** |

![Figure 3: Operational Failure Boundaries (The "Success Cliff"). Success rates for PickPlace and Stack tasks drop monotonically as the injected orientation error increases. Dotted vertical lines indicate the task-specific 50% success cliff, while dashed lines show representation-specific p99 risk. Both Atlas and Lie (fixed) exhibit risky behavior near the 30° cliff, directly predicting their empirical performance drops.](file:///c:/Users/suhas/Downloads/rotation-benchmark/rotation-benchmark/results/final_manuscript_figures/fig3_hero_cliff.png)

## 6. Empirical Validation of the Predictive Framework
The link between high-resolve tail risk ($p_{99}$) and actual task success is verified empirically in Figure 4. We observe a strong negative correlation ($\rho = -0.952$) across manipulation tasks and representations. This confirms that $p_{99}$ evaluation is a rigorous diagnostic instrument for forecasting robotic failures.

![Figure 4: Tail-Risk vs. Downstream Success. A Spearman correlation of $\rho = -0.952$ confirms that benchmark tail-risk is a direct predictor of operational robot success. Euler was not included in the downstream correlation because it was not evaluated in the manipulation study. (NutAssembly excluded—controller-limited).](file:///c:/Users/suhas/Downloads/rotation-benchmark/rotation-benchmark/results/final_manuscript_figures/fig4_predictive_correlation.png)

## 7. Conclusion
Matching benchmark tail-risk to task-specific error tolerance provides a rigorous framework for representation selection. By focusing on the $p_{99}$ metric, practitioners can eliminate the operational risk associated with minimal mappings in safety-critical robotics.
