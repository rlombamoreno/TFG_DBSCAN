# GPU-Accelerated DBSCAN

Developer: Rodrigo Lomba Moreno <br>
Institution: Universidad Politécnica de Madrid <br>
Date: November 2025 <br>
Version: 1.0 <br>

## Description
DBSCAN clustering algorithm for image segmentation and point cloud analysis with CPU and GPU implementations. Supports images (JPEG, PNG, TIFF) and NetCDF files.

## Features
- CPU implementation (NumPy + Numba)
- GPU implementation (CuPy + CUDA)
- GPU with ctypes (compiled CUDA library)
- Automatic parameter estimation
- Cluster property computation
- Visualization and histogram generation

## Requirements
CPU: Python 3.7+, NumPy, Matplotlib, Pillow, netCDF4, Numba <br>
GPU: CuPy, NVIDIA CUDA Toolkit, NVIDIA GPU (compute capability 3.0+)

## Project Structure
project/ <br>
├── src/                    # Source code <br>
├── pictures/               # Test images <br>
├── results/                # Output (CPU/GPU folders) <br>
└── Makefile                # Build system

## Installation
```bash
git clone https://github.com/yourusername/gpu-dbscan.git 
cd gpu-dbscan
```

# Install dependencies
```bash
pip install numpy matplotlib pillow netcdf4 numba cupy-cuda11x
```
# Compile CUDA library
```bash
make
```
## Command Line Options
--std_scale VALUE   : Epsilon scaling factor (0-1, default: 1.0) <br>
--min_pts VALUE     : Minimum points per cluster (default: auto) <br>
--eps VALUE         : Direct epsilon value (skips auto-calculation) <br>

## Output Files (in results/ folder)
*_clusters_*.png          : Colored cluster visualization <br>
cluster_properties_*.txt  : Cluster statistics <br>
histograms_*.png          : Distribution plots <br>
histogram_data_*.txt      : Statistical summary <br>


## Makefile Commands
make        # Compile CUDA library <br>
make clean  # Remove compiled files <br>
make info   # Show GPU compute capability <br>
make debug  # Compile with debug symbols <br>
make help   # Show help message <br>
