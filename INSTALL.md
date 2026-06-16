# Installation Guide

This guide was tested on a machine with an NVIDIA 2080Ti GPU, CUDA 11.7, and
driver 515.65.01. If your CUDA version is different, install the PyTorch and
PyTorch3D builds that match your own system.

## 0. Clone The Repository

Clone the project and enter the repository root:

```bash
git clone https://github.com/zql-kk/FlowPolicy.git
cd FlowPolicy
```

All commands below assume you are running them from the repository root unless a
step explicitly changes directories.

## 1. Create A Conda Environment

```bash
conda create -n focal python=3.8
conda activate focal
```

## 2. Install PyTorch

Install the PyTorch build that matches your CUDA version. For CUDA 11.8:

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

If you use another CUDA version, choose the corresponding installation command
from the official PyTorch selector.

## 3. Install FocalPolicy

```bash
cd FocalPolicy
pip install -e .
cd ..
```

## 4. Install MuJoCo 2.1.0

Create the MuJoCo directory if it does not already exist:

```bash
mkdir -p ~/.mujoco
cd ~/.mujoco
```

Download and extract MuJoCo:

```bash
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
  -O mujoco210.tar.gz --no-check-certificate
tar -xvzf mujoco210.tar.gz
```

Return to the repository root before continuing:

```bash
cd -
```

## 5. Configure MuJoCo Environment Variables

Add the following lines to your shell startup file, usually `~/.bashrc`:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export MUJOCO_GL=egl
```

Then reload your shell configuration:

```bash
source ~/.bashrc
```

Open a new terminal after this step to make sure the environment variables are
available everywhere.

## 6. Install `mujoco-py`

```bash
cd third_party/mujoco-py-2.1.2.14
pip install -e .
cd ../..
```

## 7. Install Simulation Dependencies

First install build tools and pinned compatibility packages:

```bash
pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0
```

Then install each local simulation package:

```bash
cd third_party/gym-0.21.0
pip install -e .
cd ../Metaworld
pip install -e .
cd ../rrl-dependencies
pip install -e mj_envs/.
pip install -e mjrl/.
cd ../..
```

## 8. Install PyTorch3D

Use one of the following options.

Option A: install the prebuilt CUDA 11.7 package:

```bash
conda install -y https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch3d/linux-64/pytorch3d-0.7.5-py38_cu117_pyt201.tar.bz2
```

Option B: install the simplified PyTorch3D package from `third_party`:

```bash
cd third_party/pytorch3d_simplified
pip install -e .
cd ../..
```

## 9. Install Other Python Packages

```bash
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf \
  hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 \
  numba==0.56.4 moviepy imageio av matplotlib termcolor natsort open3d \
  torch-dct
```

```bash
pip install huggingface_hub==0.25.2
pip install robosuite==1.5.1
pip install bddl==1.0.1
pip install future easydict
```

## 10. Optional: Install The Point Cloud Visualizer

This visualizer is provided by DP3 and is only needed if you want to inspect
point clouds.

```bash
pip install kaleido plotly
cd visualizer
pip install -e .
cd ..
```

## Quick Verification

After installation, run these checks from the repository root:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import mujoco_py; print('mujoco-py ok')"
python -c "import focal_policy_3d; print('FocalPolicy ok')"
```
