#!/bin/bash
# Reproduce Table 3: Downstream Occlusion Stress

echo "Running Occluded Pose Regression (50% occlusion)..."
python -m rotation_benchmark.downstream.occluded_pose_regression --model_path checkpoints/sixd_checkpoint.pt --rep 6D --occlusion_rate 0.5
