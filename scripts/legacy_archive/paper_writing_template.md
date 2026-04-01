# Manuscript Writing Template: Evaluating Rotation Parameterizations

This template provides a section-by-section narrative guide for the research paper, following the "Worst-Case Reliability" story.

---

### 1. Introduction

**Opening:**
*Supervised learning of 3D rotations is a fundamental component in modern robotics, with applications in object pose estimation, end-effector control, and visual odometry.*

**Transition:**
*While most benchmarks emphasize average-case metrics such as median geodesic error, these summaries can obscure rare but operationally catastrophic failures, particularly near antipodal ambiguities.*

**Closing:**
*In this work, we present a large-scale evaluation centered on worst-case reliability, showing that lower $p_{99}$ error is associated with improved downstream manipulation success in controlled robotics settings.*

---

### 2. Related Work

**Opening:**
*The choice of rotation parameterization has long been recognized as a key factor in the stability and efficiency of learning-based pose prediction.*

**Transition:**
*While continuous higher-dimensional mappings such as 6D offer favorable optimization properties, minimal representations remain attractive because of their compactness, motivating continued work on ambiguity-aware losses and geometry-aware supervision.*

**Closing:**
*We build on this literature by studying how full empirical error distributions, rather than central-tendency metrics alone, relate to downstream manipulation reliability.*

---

### 3. Methods

**Opening:**
*We compare continuous representations, minimal Lie-algebraic mappings, and chart-based multi-hypothesis models under a matched training and evaluation protocol.*

**Transition:**
*To probe whether loss design can mitigate antipodal instability in minimal mappings, we introduce a FullFix variant that combines ambiguity-aware supervision with architectural hardening.*

**Closing:**
*All methods are evaluated using geodesic error, with particular emphasis on tail metrics such as $p_{99}$ and failure rate above $30^{\circ}$.*

---

### 4. Main Benchmark Results

**Opening:**
*Across a 25-million-sample synthetic benchmark, low-tail continuous methods form the clear high-reliability tier under worst-case evaluation.*

**Transition:**
*Although median errors are often relatively close across methods, tail behavior differs sharply, with several compact alternatives exhibiting much heavier $p_{99}$ error.*

**Closing:**
*These results show that average accuracy alone is insufficient for judging representation reliability in robotics-relevant settings.*

---

### 5. Failure Anatomy

**Opening:**
*Error stratified by ground-truth rotation magnitude reveals that failures are concentrated near the antipodal boundary rather than uniformly distributed across SO(3).*

**Transition:**
*Ablation results suggest that ambiguity-aware objectives and architectural hardening can reduce tail failures, though they do not fully remove the fragility of minimal mappings in the hardest regimes.*

**Closing:**
*This identifies the hard-angle region near $\pi$ as the dominant source of catastrophic rotation failure.*

---

### 6. Downstream Robotics Validation

**Opening:**
*To study the operational consequences of representation-specific error patterns, we evaluate both controlled error injection and live model-in-the-loop performance in robosuite.*

**Transition:**
*Percentile-based perturbation sweeps reveal a practical “cliff zone” in which task success degrades sharply once errors enter the upper tail of a method’s empirical distribution.*

**Closing:**
*Across achievable tasks, low-tail methods maintain high success, while higher-tail alternatives degrade substantially, linking benchmark tail risk to downstream manipulation reliability.*

---

### 7. Limitations and Conclusion

**Opening:**
*Our experiments provide strong evidence that tail-error metrics such as $p_{99}$ are informative predictors of downstream reliability in controlled supervised rotation learning settings.*

**Transition:**
*At the same time, the current benchmark backbone is point-order dependent, and transfer to arbitrary point sets or mesh scans will likely require permutation-invariant architectures such as PointNet-style or transformer-based models.*

**Closing:**
*For robotics practitioners, these findings support using low-tail continuous representations as strong default choices when worst-case reliability matters.*
