# Learning 3D Vision

To learn and follow state-of-the-art updates about 3D computer vision, this repo features useful links to wonderful tutorials, slides, courses, papers and videos and so on. I will make updates regularly.

## ⚙️ Setup

### Conda Environment

Our code is developed based on pytorch 2.6.0, CUDA 12.6 and python 3.10.

We recommend using conda for installation:
```bash
# we start with 3DGS repo setup
conda create -y -n 3dvGS python=3.10
conda activate 3dvGS

# Pytorch 2.6.0, CUDA 12.6
#pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Pay attention to the "git+https://github.com/dcharatan/diff-gaussian-rasterization-modified" lib, you might fail due to CUDA version and G++/GCC version. Please check the [CUDA-GCC compatibility table](https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version).


```bash
# to install this diff-gaussian-rasterization-modified
# make sure your cuda version and g++, gcc version matched
# see: https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version
# CUDA version + max supported GCC version
# e.g., cuda 12.4, 12.5, 12.6, with GCC <= 13.2
# e.g., cuda 12.8,  with GCC <= 14
git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
```

With CUDA 12.6, my previous G++ is 

```plain
g++ (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
```
cannot support each other. So I downgrade the G++ to G++12 as:

```bash
# 1) Downgrade GCC: Install GCC 12 or another version known to be compatible with CUDA 12.6. 
# On Ubuntu, you can install GCC 12 using:​
sudo apt-get update
sudo apt-get install gcc-12 g++-12

# 2) After installation, you can switch the default GCC and g++ versions 
# using the update-alternatives tool:
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# 3) Then, select the default version:
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```
After this G++/GCC version adjustment, the `pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified` works.

## 🔥 Neural Rendering

### Neural Radiance Fields - NeRF 

### 3D Gaussian Splatting - 3DGS