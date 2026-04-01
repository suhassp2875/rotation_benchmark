import robosuite as suite
import numpy as np
from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
import PIL.Image as Image
import os

def test_render():
    os.makedirs("results/images", exist_ok=True)
    
    arm_controller_config = load_part_controller_config(default_controller="OSC_POSE")
    controller_config = refactor_composite_controller_config(arm_controller_config, "Panda", ["right"])
    
    env = suite.make(
        "PickPlace",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        controller_configs=controller_config,
    )
    
    obs = env.reset()
    img = obs["agentview_image"]
    # robosuite images are upside down due to mujoco convention
    img = np.flipud(img)
    
    pil_img = Image.fromarray(img)
    pil_img.save("results/images/robosuite_test.png")
    print("Screenshot saved to results/images/robosuite_test.png")
    env.close()

if __name__ == "__main__":
    try:
        test_render()
    except Exception as e:
        print(f"Error testing robosuite: {e}")
