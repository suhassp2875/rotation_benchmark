# Migration Guide: Windows to Ubuntu 22.04

Moving your research environment to Linux is highly recommended for robotics work, but there are several "non-portable" components you need to handle manually.

## 1. Zip Checklist
You only need to zip your primary research folder. Since your `robosuite` install is a clean clone of the official repository, you can simply reinstall it on Ubuntu.

| Component | Status | Action |
| :--- | :--- | :--- |
| **Project Folder** | ✅ Essential | Zip `rotation-benchmark/` in its entirety. |
| **Robosuite** | 🟢 Reinstall | Do **NOT** zip. Run `pip install robosuite` on Ubuntu. |
| **Python Env** | ❌ Skip | Do **NOT** zip `.venv` or Conda folders. |
| **BOP Datasets** | 💾 Optional | `bop/ycbv` is large. Re-download on Ubuntu if you have fast internet. |
| **Checkpoints** | ✅ Essential | Ensure `outputs/` and `checkpoints/` are in the project zip. |

## 2. Environment Reconstruction
Since Windows binaries won't work on Ubuntu, you must recreate your environment:

1. **Install Miniconda/Anaconda** on Ubuntu.
2. **Re-create the environment**:
   ```bash
   conda create -n rotation python=3.9
   conda activate rotation
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install robosuite
   # See requirements.yml for full list
   ```
3. **Paths**: Update any hardcoded `C:\` paths in your scripts to Unix format (e.g., `/home/suhas/rotation-benchmark`).

## 3. Ubuntu-Specific Robotics Setup
Robosuite and MuJoCo require specific system libraries on Ubuntu 22.04:

```bash
# Install GL dependencies for rendering
sudo apt update
sudo apt install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev
sudo apt install mesa-utils patchelf

# Set up MuJoCo path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

## 4. Migration Steps
1. **Windows**: Run `conda env export > environment.yml` inside your project folder.
2. **Windows**: Zip the two folders (`rotation-benchmark` and `robosuite`).
3. **Ubuntu**: Unzip to your home directory.
4. **Ubuntu**: Run `conda env create -f environment.yml`.
5. **Ubuntu**: Install `robosuite` in editable mode if you modified the source:
   ```bash
   cd ~/robosuite
   pip install -e .
   ```

> [!IMPORTANT]
> **Check your Git state**: Ensure all changes are committed and pushed to a remote branch before you wipe the Windows drive!
