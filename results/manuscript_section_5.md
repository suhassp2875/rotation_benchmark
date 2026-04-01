# Section 5: The Anatomy of Geometric Failure

The failures observed in both our benchmark (Section 4) and robotic trials (Section 6) are not random; they are localized to specific "singular" regimes of the rotation space. In this section, we analyze the relationship between representation topology and operational stability.

### 5.1 Geometric Ambiguity is not Gradient Instability
A common misconception is that minimal representations fail because of "bad gradients" at the singularity. However, our stratified analysis shows that even with perfectly optimized weights (as seen in our Safe region parity), minimal mappings like Euler angles and Quaternions cannot escape the costs of their underlying topology. 

In the **Antipodal region**, the ground-truth rotation $\pi$ is exactly equidistant from $R$ and its inverse in several minimal mappings, or represents the point where the mapping must "flip" (e.g., $q$ to $-q$). This creates a **Discontinuity in the Output Space**.

### 5.2 From Discontinuity to Controller Instability
For an Operational Space Controller (OSC), a small change in predicted orientation should result in a small corrective torque. However, near an antipodal discontinuity, a negligible shift in the input point cloud can cause the model to "jump" between distant points in the representation space (e.g. from $+\pi$ to $-\pi$). 

When this happens model-in-the-loop:
1.  **Torque Spikes**: The controller perceives a massive orientation error and commands maximal corrective torque.
2.  **Mechanical Instability**: In our *robosuite* evaluations, this manifested as violent "wrist-flip" motions and table collisions, even when the average $L_2$ error remained low.

### 5.3 Benchmarking the Discontinuity
This explains why $p_{99}$ and our proposed **Discontinuity Metric** are such powerful predictors of the Success Cliff. While the median error averages out these rare "jumps," the tail-risk captures the exact events that trigger controller divergence. Methods that maintain continuity across all of SO(3) (such as **6D** and **SVD**) are immune to this specific failure mode, explaining their flat curves in Figure 1.
