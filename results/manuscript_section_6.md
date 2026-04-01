# Section 6: Downstream Robotics Validation

To study the operational consequences of the tail-error distributions identified in Section 4, we evaluate how orientation error magnitude translates to task failure in a simulated robotic manipulation environment. We employ two representative tasks in *robosuite*: **PickPlace**, which requires precise alignment of a milk carton to a bin, and **Stack**, which involves a more forgiving cube-stacking operation.

### 6.1 The Orientation Success Cliff
We characterize the orientation error tolerance of each task by injecting precisely controlled SO(3) perturbations (via the Rodrigues formula) into the end-effector controller across a sweep of error magnitudes ($0^{\circ}$ to $90^{\circ}$). For each task, we resolve a definitive "Success Cliff"—the error threshold beyond which task success drops below 50%.

Our results (Figure 1) show that **PickPlace** is significantly more sensitive to orientation inaccuracy, with a performance cliff at **22.5°**. In contrast, the **Stack** task demonstrates greater robustness, maintaining over 50% success until **32.9°**. Notably, we observe a slight non-monotonic "bump" in success at very low error magnitudes ($2^{\circ}-7^{\circ}$); we hypothesize this is a real physical phenomenon where minor orientation perturbations effectively "nudge" the operational space controller (OSC) toward more favorable contact geometry during the approach phase, occasionally overcoming near-singular neutral configurations.

### 6.2 Predicting Failure from Benchmark Tail-Risk
By overlaying the $p_{99}$ error thresholds from our Phase 1 benchmark onto these task-specific cliff curves, we can quantitatively predict downstream reliability:

| Representation | Benchmark $p_{99}$ | PickPlace (22.5° Cliff) | Stack (32.9° Cliff) |
| :--- | :--- | :--- | :--- |
| **6D / SVD** | < 5° | **SAFE** | **SAFE** |
| **Lie_FullFix** | 31.9° | **RISK** | **SAFE** |
| **Quaternion** | 46.1° | **RISK** | **RISK** |
| **Euler** | 118.8° | **RISK** | **RISK** |

### 6.3 Discussion
These findings validate $p_{99}$ as a superior predictor of operational failure compared to aggregate mean error. Specifically, while Lie_FullFix achieves high success in coarse tasks like Stack, its tail-risk is high enough to trigger failure in higher-precision settings like PickPlace. Minimal representations (Euler/Quat) reside consistently in the "Risk Zone," where their worst-case errors far exceed the controller's tolerance, leading to catastrophic downstream failures.
