# Mathematical Foundation: Rotation Learning Reliability

This document formalizes the problem setup, evaluation metrics, and the topological challenges associated with rotation regression, providing the mathematical backbone for our empirical results.

## 1. Problem Formulation

We define the supervised rotation regression problem as learning a mapping from an input space $\mathcal{X}$ to the 3D rotation group $SO(3)$. Given an input $x \in \mathcal{X}$ (e.g., a point cloud) and a neural network $f_\theta$, we predict a parameterization $z$:

$$ f_\theta(x) \to z, \quad \hat{R} = \mathrm{decode}(z) $$

The error between the predicted rotation $\hat{R}$ and the true rotation $R$ is measured using the angular geodesic distance:

$$ d(R, \hat{R}) = \arccos\left(\frac{\mathrm{tr}(R^\top \hat{R}) - 1}{2}\right) $$

## 2. Reliability Metrics & Tail Risk

While standard empirical evaluations report the mean or median of $d(R, \hat{R})$, these central tendencies are blind to catastrophic but rare failures ("tail risk"). To characterize worst-case reliability, we formally define the $99^{\mathrm{th}}$ percentile error:

$$ q_{0.99} = \inf \{ t : \Pr(d(R, \hat{R}) \le t) \ge 0.99 \} $$

Additionally, we define a strict failure rate for a catastrophic error threshold (e.g., $30^\circ$):

$$ \mathrm{Fail}_{30} = \Pr(d(R, \hat{R}) > 30^\circ) $$

These metrics directly penalize representations that suffer from rare but severe outliers.

## 3. The Antipodal Ambiguity

The primary source of tail risk in rotation learning is the topology of $SO(3)$. Specifically, around the antipodal boundary (rotation angles near $\pi$), representations experience mathematical stress.

**Observation 1 (Antipodal Ambiguity):** *Near rotation angle $\pi$, minimal Euclidean parameterizations (such as axis-angle or quaternions) admit multiple or discontinuous target encodings for the same (or nearly indistinguishable) physical rotations. This creates unstable or conflicting supervision under standard regression losses.*

### Gated Loss Formalization (Lie_FullFix)
To mitigate this instability for Lie algebra parameterizations (axis-angle vectors $w = \theta u$), we use a gated loss mechanism. The localized loss is weighted such that Euclidean gradients are naturally muted near the ambiguity:

$$ \mathcal{L}_{total} = \mathcal{L}_{geo} + \lambda_{axis} \mathcal{L}_{axis} $$
$$ \mathcal{L}_{axis} = \sin^2(\theta_{gt}) (1 - u_{pred} \cdot u_{gt}) + \text{ramp}(\theta_{gt}) (1 - |u_{pred} \cdot u_{gt}|) $$

The $\sin^2(\theta_{gt})$ envelope smoothly suppresses conflicting axis supervision as $\theta_{gt} \to \pi$, while the absolute dot product term ($1 - |u \cdot v|$) symmetrically handles the $u \approx -u$ ambiguity near the boundary.

## 4. Downstream Task Success Link

Our empirical evaluations aim to demonstrate that synthetic $q_{0.99}$ tail risk strongly predicts downstream robotic manipulation success. We model task success probability conditioned on an orientation error $\delta = d(R, \hat{R})$ as:

$$ S(\delta) = \Pr(\text{task succeeds} \mid d(R, \hat{R}) = \delta) $$

**Empirical Hypothesis:** *There exists a controller- and task-dependent basin in which success $S(\delta)$ remains high for small orientation errors, but drops sharply beyond a task-specific threshold $\delta_{\text{crit}}$.*

By injecting controlled orientation perturbations $\delta$ derived from our phase-one percentile sweeps (e.g., drawing $\delta = q_{0.99}^{\text{model\_A}}$) into downstream simulations (Robosuite), we empirically estimate this relationship. This protocol isolates representation-induced failure modes from general perception noise.
