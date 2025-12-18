# GPU-Accelerated DBSCAN

Developer: Rodrigo Lomba Moreno
Institution: Universidad Politécnica de Madrid
Date: November 2025
Version: 1.0

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
CPU: Python 3.7+, NumPy, Matplotlib, Pillow, netCDF4, Numba
GPU: CuPy, NVIDIA CUDA Toolkit, NVIDIA GPU (compute capability 3.0+)

## Project Structure
project/
├── src/                    # Source code
├── pictures/               # Test images
├── results/                # Output (CPU/GPU folders)
└── Makefile                # Build system

## Installation
git clone https://github.com/yourusername/gpu-dbscan.git
cd gpu-dbscan

# Install dependencies
pip install numpy matplotlib pillow netcdf4 numba cupy-cuda11x

# Compile CUDA library
make

## Command Line Options
--std_scale VALUE   : Epsilon scaling factor (0-1, default: 1.0)
--min_pts VALUE     : Minimum points per cluster (default: auto)
--eps VALUE         : Direct epsilon value (skips auto-calculation)

## Output Files (in results/ folder)
*_clusters_*.png          : Colored cluster visualization
cluster_properties_*.txt  : Cluster statistics
histograms_*.png          : Distribution plots
histogram_data_*.txt      : Statistical summary

## Performance
CPU: Best for small images (<1 MP) or when no GPU is available
GPU: 5–50x faster for large images (>1 MP)
Requirements: NVIDIA GPU with CUDA support

## Makefile Commands
make        # Compile CUDA library
make clean  # Remove compiled files
make info   # Show GPU compute capability
make debug  # Compile with debug symbols
make help   # Show help message

## Troubleshooting
CUDA library not found:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)

Memory errors:
Use CPU version for very large images

Slow CPU:
Ensure Numba is installed

## License
MIT License

## Contact
For issues or questions: rlomba@example.com
