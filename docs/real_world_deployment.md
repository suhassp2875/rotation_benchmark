# Real-World Deployment Strategy: uArm Swift Pro

This guide outlines the path for transitioning our simulation-trained rotation models to the physical **uArm Swift Pro**. It prioritizes safety, hardware constraints, and a tiered implementation order.

## 1. System Architecture
Separate the "intelligence" from the "execution" to maintain low latency and high reliability.

- **PC Side (Ubuntu/Linux)**: Runs the `.pt` models (Inference), processes vision data, and computes target coordinates.
- **Hardware Side (uArm Swift Pro)**: Strictly for execution using the **uArm Python SDK** (SwiftAPI).

## 2. Hardware Constraints & Considerations
The uArm Swift Pro is a **4-DOF** tabletop arm. It is optimized for **horizontal Pick-and-Place**, not full-pose 6D manipulation.

| Parameter | Constraint | Deployment Impact |
| :--- | :--- | :--- |
| **DOF** | 4-DOF (XYZ + Wrist Yaw) | Model must provide **targets**, not raw actuation. |
| **Range** | 50–320 mm | Workspace must be bounded in software. |
| **Precision** | 0.2 mm | Highly repeatable; failures are usually due to calibration/perception. |
| **Wrist** | 0–180° | Wrist behavior varies with gripper attachment; must be verified physically. |

## 3. Tiered Implementation Path

### Phase 1: Hardware Baseline (No ML)
Ensure the arm is mechanically reliable before adding complexity.
1. Connect via USB and initialize `SwiftAPI`.
2. Move to a safe `HOME` pose and verify `set_mode(3)` (Gripper Mode).
3. Perform a hard-coded Pick-and-Place loop to confirm repeatability.

### Phase 2: Planar Perception
Convert pixels to robot coordinates.
1. Mount a fixed overhead camera.
2. Use **Planar Calibration (Homography)** to map image $(u, v)$ to robot $(x, y)$.
3. Depth is optional for flat-table tasks; image-based centroids are sufficient for initial success.

### Phase 3: Model Integration
Introduce our rotation-reliable models (`SVD`, `6D`) as **target estimators**.
1. **Model Role**: Output the object's in-plane orientation ($\theta$) or 3D pose.
2. **Robot Role**: Execute a scripted trajectory (Hover $\rightarrow$ Descend $\rightarrow$ Grasp $\rightarrow$ Lift).
3. **Task Projection**: Use the model's prediction to set the wrist angle (`set_wrist`), but keep Pitch/Roll fixed by design.

## 4. Software Implementation Template

```python
from uarm.wrapper import SwiftAPI
import time

swift = SwiftAPI(port='/dev/ttyACM0') # Ubuntu standard
swift.waiting_ready()
swift.set_mode(3) # Gripper/Pen mode

# Safety Coordinates
HOME = {'x': 150, 'y': 0, 'z': 150}
HOVER_Z = 100
PICK_Z = 35 # Adjusted for your gripper height

# 1. Perception Step (Example only)
# pick_x, pick_y, theta = your_model.predict(camera_frame)

# 2. Execution Sequence
swift.set_position(**HOME, wait=True)
swift.set_wrist(theta, wait=True)           # Verify physically first
swift.set_gripper(catch=False, wait=True)   # Open

swift.set_position(x=pick_x, y=pick_y, z=HOVER_Z, wait=True)
swift.set_position(x=pick_x, y=pick_y, z=PICK_Z, wait=True)
swift.set_gripper(catch=True, wait=True)    # Close
time.sleep(0.5)

swift.set_position(z=HOVER_Z, wait=True)    # Retreat
```

## 5. Critical Safety Rules
- **Verify before Action**: Run a "Static Test" where the model draws its prediction on the screen before the arm moves.
- **Safety Box**: Define limits for `x, y, z` to prevent collisions with the table or the arm’s own base.
- **Emergency Stop**: Keep a keyboard interrupt (`Ctrl+C`) or a physical disconnect ready for the first autonomous trials.
