# Section 4: Main Benchmark Results

We first evaluate the representations across a large-scale synthetic benchmark (N=25,000,000) to characterize their fundamental error distributions. To isolate the failure modes unique to each parameterization, we stratify results into three geometric regimes: **Safe** (central SO(3)), **Gimbal** (near representational singularities), and **Antipodal** (near-$\pi$ rotations).

### 4.1 Safe-Region Equivalence (The Control)
In the thermally-stable regions of SO(3), all representations perform nearly identically. Median geodesic errors range from **1.63° (SVD)** to **2.44° (Quaternion)**, with consistent $p_{99}$ metrics below **5°** (excluding Lie_FullFix). This reinforces that no single representation has a fundamental advantage in average-case learning; the choice of mapping only manifests at the distribution's edges.

### 4.2 Stratified Tail Divergence
The reliability story changes sharply in restricted geometric regimes (Table 1). While **6D** and **SVD** maintain a flat error profile across all of SO(3), minimal and compact representations exhibit catastrophic tail expansion:

*   **Euler Angles**: While accurate in safe regions ($p_{99}=3.8°$), its error explodes to **99.1°** in the Gimbal region and **165.2°** in Antipodal regimes.
*   **Quaternions**: Exhibits a massive $p_{99}$ jump to **57.7°** in Antipodal regions, driven by the $\pm q$ ambiguity.
*   **Lie_FullFix**: Shows improved stability over raw Euler, but still retains a $p_{99}$ tail of **45.0°** in Antipodal zones.

### 4.3 Predictive Link to Task Success
Crucially, these benchmark metrics are not just theoretical artifacts; they directly "shadow" the operational success thresholds established in Section 6. Specifically, the **29.6° p99** of Quaternions in the Gimbal region already places it at the very edge of the **22.5° PickPlace cliff**, predicting the high failure rates observed when grasping objects at extreme orientations. This stratification provides the "measurement instrument" necessary to forecast robotic failure without exhaustive downstream simulation.

---

# Final Paper Outline (Representative Analysis Framing)

1.  **Introduction**: Moving from average accuracy to worst-case reliability.
2.  **Related Work**: Representation geometry and robotic stability.
3.  **The Predictive Benchmark**: Stratified evaluation of SO(3) representations.
4.  **Benchmark Results**: (Section 4) Safe-region parity vs. Tail-region divergence.
5.  **Failure Anatomy**: The link between input-space ambiguity and output-space discontinuity.
6.  **Downstream Validation**: (Section 6) Resolving task "Success Cliffs" and validating benchmark risk margins.
7.  **Discussion and Conclusion**: Practical guidance—matching benchmark $p_{99}$ to task tolerance.
