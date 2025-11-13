"""
gpu_dbscan_ctypes.py
--------------------
GPU implementation of the DBSCAN algorithm using CuPy and CUDA kernels with ctypes.

Author: Rodrigo Lomba Moreno
Institution: Universidad Politécnica de Madrid  
Date: November 2025  
Version: 1.0

Description:
This script implements the DBSCAN algorithm for image segmentation using
CuPy for GPU acceleration and CUDA kernels with ctypes for the core algorithm.
It supports input from common image files (JPEG/PNG) and NetCDF files.

Usage:
    python3 gpu_dbscan_ctypes.py <input_filename> [std_scale] [min_pts]

    - <input_filename> : Path to an image (.jpg, .png) or a NetCDF (.nc) file.
    - [std_scale]      : Optional float in [0, 1] used to scale the std in the
                         epsilon heuristic. If omitted, std_scale defaults to 1.0.
    - [min_pts]        : Optional integer for minimum points parameter.
                         If omitted, calculated as 2*dimension + 1.

Dependencies:
    - cupy
    - numpy
    - pillow
    - matplotlib
    - netCDF4
    - ctypes
    - time, os, sys
"""

import sys
import numpy as np
import cupy as cp
import ctypes
import matplotlib.pyplot as plt
import netCDF4 as nc
from PIL import Image
from numba import jit
import time
import os


# ---------------------------
# Global constants
# ---------------------------
THREADS_PER_BLOCK = 256

#---------------------------
#Library loading
#---------------------------
def load_cuda_library():
    """
    Load the compiled CUDA library and define function signatures.
    
    Returns:
        ctypes.CDLL: Loaded library object
    """
    lib = ctypes.CDLL('./src/libdbscan.so')

    # Define function signatures
    lib.dbscan_core_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # points
        ctypes.POINTER(ctypes.c_int),    # labels
        ctypes.POINTER(ctypes.c_int),    # vector_degree
        ctypes.POINTER(ctypes.c_int),    # vector_type
        ctypes.POINTER(ctypes.c_int),    # adjacent_indexes
        ctypes.POINTER(ctypes.c_int),    # adjacent_list
        ctypes.c_int,                    # min_pts
        ctypes.c_int,                    # num_points
        ctypes.c_int                     # adjacent_list_size
    ]
    lib.dbscan_core_cuda.restype = ctypes.c_int
    
    return lib

# Load library globally
lib = load_cuda_library()


# ---------------------------
# Parameter loading functions
# ---------------------------
def load_parameters():
    """
    Load optional parameters from command line arguments.

    Returns:
        tuple: (std_scale: float, min_pts: int)
    """
    std_scale = 1.0  # default value
    min_pts = None   # will be calculated based on dimension
    
    # Parse command line arguments
    if len(sys.argv) >= 3:
        try:
            std_scale = float(sys.argv[2])
            if std_scale < 0 or std_scale > 1:
                print("gpu_dbscan: std_scale must be between 0 and 1")
                sys.exit(1)
            print(f"gpu_dbscan: Using user-provided std_scale: {std_scale}")
        except ValueError:
            print("gpu_dbscan: std_scale must be a float between 0 and 1")
            sys.exit(1)
    else:
        print("gpu_dbscan: Using default std_scale: 1.0")
    
    if len(sys.argv) >= 4:
        try:
            min_pts = int(sys.argv[3])
            if min_pts < 1:
                print("gpu_dbscan: min_pts must be a positive integer")
                sys.exit(1)
        except ValueError:
            print("gpu_dbscan: min_pts must be a positive integer")
            sys.exit(1)
    
    return std_scale, min_pts


def calculate_min_pts(user_min_pts=None):
    """
    Calculate min_pts parameter. If user provided, use that value.
    Otherwise, calculate as 2 * dimension + 1.
    For image analysis, dimension is always 2.
    """
    if user_min_pts is not None:
        print(f"gpu_dbscan: Using user-provided min_pts: {user_min_pts}")
        return user_min_pts
    
    # For image analysis, dimension is always 2 (x, y coordinates)
    dimension = 2
    calculated_min_pts = 2 * dimension + 1
    print(f"gpu_dbscan: Calculated min_pts as 2 * dimension + 1 => 2 * {dimension} + 1 = {calculated_min_pts}")
    return calculated_min_pts


def load_data():
    """
    Load points from an image (.jpg/.png) or NetCDF (.nc) file.

    Returns:
        tuple: (points: cp.ndarray of shape (N*2,), std_scale: float, min_pts: int)
    """
    if len(sys.argv) < 2:
        print("gpu_dbscan: Must specify one supported file, either image or netCDF")
        print("example: python3 script.py <filename> [std_scale] [min_pts]")
        sys.exit(1)
    
    std_scale, user_min_pts = load_parameters()
    filename = sys.argv[1]
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == ".jpg" or ext == ".png":
        points = load_image(filename)
    elif ext == ".nc":
        points = load_netcdf(filename)
    else:
        print(f"gpu_dbscan: Unsupported file extension: {ext}")
        sys.exit(1)
    
    # Calculate min_pts (always 2D for images)
    min_pts = calculate_min_pts(user_min_pts)
    
    return points, std_scale, min_pts


# ---------------------------
# Data loading functions
# ---------------------------
def load_image(image_filename):
    """
    Convert a binary image to a list of points corresponding to the cluster color.

    Parameters:
        image_filename (str): Path to image file

    Returns:
        cp.ndarray: Flattened array of shape (N*2,) with coordinates of cluster points
    """
    image_orig = Image.open(image_filename)
    print(f"gpu_dbscan: Image name: {image_filename} Size: {image_orig.size}")
    
    image_bw = cp.array(image_orig.convert('1'), dtype=int)
    hist = compute_histogram(image_bw)
    # color_marker indicates the background color; get_points_in_cluster extracts only foreground pixels
    color_marker = 1 if hist[0] < hist[1] else 0
    
    points = get_points_in_cluster(image_bw, color_marker)
    
    points = points.reshape(-1, 2)
    points = points[cp.lexsort(cp.stack((points[:,1], points[:,0])))]
    points = points.ravel()
    return points


def load_netcdf(nc_filename):
    """
    Load 2D points from a NetCDF file containing coordinates.

    Parameters:
        nc_filename (str): Path to NetCDF file

    Returns:
        cp.ndarray: Flattened array of shape (N*2,) with coordinates
    """
    ncdata = nc.Dataset(nc_filename)
    frame = 1
    atoms = ncdata.variables['coordinates'][:][frame]
    ncdata.close()
    
    if np.ma.isMaskedArray(atoms):
        atoms = np.ma.filled(atoms, fill_value=np.nan)
    
    r = atoms.transpose()
    points = points_to_array(r)
    return points

def points_to_array(r):
    """
    Convert a 2xN coordinate array to flattened N*2 float32 array.

    Parameters:
        r (np.ndarray): 2xN array

    Returns:
        cp.ndarray: Flattened array of shape (N*2,) of points
    """
    num_points = r.shape[1]
    points = cp.zeros(num_points * 2, dtype=cp.float32)
    r_gpu = cp.array(r[:2, :], dtype=cp.float32,order='C')
    points[0::2] = r_gpu[0, :]
    points[1::2] = r_gpu[1, :]
    return points


def save_image(image_colored):
    """
    Save the clustered image to disk.
    
    Parameters:
        image_colored (np.ndarray): RGB image with clusters colored
    """
    image_filename = sys.argv[1]
    name, ext = os.path.splitext(image_filename)
    output_filename = f"{name}_clusters_GPU.png"
    plt.imsave(output_filename, image_colored)
    print(f"gpu_dbscan: Clustered image saved as: {output_filename}")


# ---------------------------
# Image utility functions
# ---------------------------
def compute_histogram(image):
    """
    Compute histogram of a binary image.
    """
    y_len, x_len = image.shape
    color_histogram = cp.histogram(image, bins=[0, 1, 2])[0]
    return color_histogram


def count_cluster_points(image, color_marker, y_len, x_len):
    """
    Count the number of foreground pixels in the image.
    
    Parameters:
        image (cp.ndarray): Binary image
        color_marker (int): Background color value
        y_len (int): Image height
        x_len (int): Image width
        
    Returns:
        int: Number of foreground pixels
    """
    count = cp.zeros(1, dtype=cp.int32)
    
    blocks_per_grid = (y_len * x_len + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    
    kernel_code =  r'''
    extern "C" __global__
    void count_points(const int *image, int *count, const int color_marker, const int y_len, const int x_len) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < y_len * x_len) {
            int x = idx % x_len;
            int y = idx / x_len;
            int linear_idx = y * x_len + x;
            if (image[linear_idx] != color_marker) {
                atomicAdd(&count[0], 1);
            }
        }
    }
    '''
    module = cp.RawModule(code=kernel_code)
    count_kernel = module.get_function('count_points')
    count_kernel((blocks_per_grid,), (THREADS_PER_BLOCK,), (image.ravel().astype(cp.int32), count, color_marker, y_len, x_len))
    # Transfer the result to host and return as Python int
    return int(count.get()[0])


def get_points_in_cluster(image, color_marker):
    """
    Extract coordinates of all foreground pixels using CUDA kernel.
    
    Parameters:
        image (cp.ndarray): Binary image
        color_marker (int): Background color value
        
    Returns:
        cp.ndarray: Array of foreground point coordinates
    """
    y_len, x_len = image.shape
    count = count_cluster_points(image, color_marker, y_len, x_len)

    # count is an integer scalar now
    if count == 0:
        return cp.empty((0,2), dtype=cp.int32)

    points_index = cp.zeros(1, dtype=cp.int32)
    
    points = cp.zeros(count * 2, dtype=cp.float32)
    
    blocks_per_grid = (y_len * x_len + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    
    kernel_code = r'''
    extern "C" __global__
    void get_points_in_cluster(const int *image, float *points, const int color_marker, const int y_len, const int x_len, int *points_index) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < y_len * x_len) {
            int x = idx % x_len;
            int y = idx / x_len;
            int linear_idx = y * x_len + x;
            if (image[linear_idx] != color_marker) {
                int point_idx = atomicAdd(&points_index[0], 1);
                points[point_idx * 2]     = float(x);
                points[point_idx * 2 + 1] = float(y);
            }
        }
    }
    '''
    module = cp.RawModule(code=kernel_code)
    get_points_kernel = module.get_function('get_points_in_cluster')
    get_points_kernel((blocks_per_grid,), (THREADS_PER_BLOCK,), (image.ravel().astype(cp.int32), points, color_marker, y_len, x_len, points_index))

    # Get the actual number of points written by the kernel
    num_points = int(points_index.get()[0])
    if num_points == 0:
        return cp.empty((0,2), dtype=cp.float32)

    return points


# ---------------------------
# Epsilon calculation functions
# ---------------------------
def get_epsilon(points, k,std_scale):
    """
    Compute the adaptive epsilon using k-distances and standard deviation.

    Parameters:
        points (cp.ndarray): Flattened array of points
        min_pts (int): Number of neighbors for k-distance
        std_scale (float): Scaling factor for standard deviation

    Returns:
        cp.ndarray: Recommended epsilon value
    """
    kn_distances = compute_kn_distances(points, k) # k-distances
    # Heuristic: epsilon = mean + std_dev * std_scale
    epsilon = cp.zeros(1, dtype=cp.float32)
    epsilon[0]= cp.mean(kn_distances) + cp.std(kn_distances) * std_scale

    print(f"gpu_dbscan: Recommended epsilon: {epsilon[0]}")
    return epsilon

def compute_kn_distances(points, k):
    """
    Compute the k-nearest distances for each point using CUDA kernel.

    Parameters:
        points (cp.ndarray): Flattened array of points
        k (int): Number of neighbors

    Returns:
        cp.ndarray: Array of k-distances for each point
    """
    num_points = len(points) // 2
    kn_distances = cp.empty(num_points, dtype=cp.float32)

    blocks_per_grid = (num_points + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    
    MAX_K = k
    kernel_code =  f'''
    #define MAX_K {MAX_K} // Define MAX_K based on k, needed for array size
    
    extern "C" __global__
    void compute_kn_distances(const float *points, float *kn_distances, const int num_points, const int k) {{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_points) {{
            float x1 = points[idx * 2];
            float y1 = points[idx * 2 + 1];
            float min_dists[MAX_K];
            for (int i = 0; i < k; i++) {{
                min_dists[i] = 1e20f; // Valor grande
            }}

            for (int j = 0; j < num_points; j++) {{
                if (j != idx) {{
                    float x2 = points[j * 2];
                    float y2 = points[j * 2 + 1];
                    float dx = x1 - x2;
                    float dy = y1 - y2;
                    float dist = (dx * dx + dy * dy);
                    if (dist < min_dists[k-1]) {{
                        for (int pos = 0; pos < k; pos++) {{
                            if (dist < min_dists[pos]) {{
                                // Desplazar y insertar
                                for (int m = k-1; m > pos; m--) {{
                                    min_dists[m] = min_dists[m-1];
                                }}
                                min_dists[pos] = dist;
                                break;
                            }}
                        }}
                    }}
                }}
            }}
            kn_distances[idx] = sqrtf(min_dists[k-1]);
        }}
    }}
    '''
    module = cp.RawModule(code=kernel_code)
    compute_kn_distances_kernel = module.get_function('compute_kn_distances')
    compute_kn_distances_kernel((blocks_per_grid,), (THREADS_PER_BLOCK,), (points, kn_distances, num_points, k))
    return kn_distances


# ---------------------------
# Graph construction functions
# ---------------------------
def build_graph(points, eps,min_pts):
    """
    Build the neighborhood graph for DBSCAN.

    Parameters:
        points (cp.ndarray): Flattened array of points
        eps (cp.ndarray): Epsilon value
        min_pts (int): Minimum points for core point

    Returns:
        tuple: (vector_degree, vector_type, adjacent_indexes, adjacent_list)
    """
    num_points = len(points) // 2
    
    vector_degree = cp.zeros(num_points, dtype=cp.int32) # Vertices degree
    vector_type = cp.zeros(num_points, dtype=cp.int32) # Vertex type: core or not
    vector_degree,vector_type = neigbours_count(points, eps, vector_degree, vector_type,min_pts)
    
    # Calcular adjacent_indexes usando prefix sum excluyente
    adjacent_indexes = cp.zeros(num_points, dtype=cp.int32)
    if num_points > 1:
        adjacent_indexes[1:] = cp.cumsum(vector_degree[:-1])
    
    total_neighbors = int(vector_degree.sum())
    
    adjacent_list = build_adjacency_list_from_indexes(points, eps, vector_degree, adjacent_indexes)
    
    return vector_degree, vector_type, adjacent_indexes, adjacent_list


def neigbours_count(points, eps, vector_degree, vector_type, min_pts):
    """
    Count neighbors within epsilon for each point and identify core points.

    Parameters:
        points (cp.ndarray): Flattened array of points
        eps (cp.ndarray): Epsilon value
        vector_degree (cp.ndarray): Array to store neighbor counts
        vector_type (cp.ndarray): Array to mark core points
        min_pts (int): Minimum points for core point

    Returns:
        tuple: (vector_degree, vector_type)
    """
    num_points = len(points) // 2
    
    blocks_per_grid = (num_points + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    
    kernel_code =  f'''
    
    extern "C" __global__
    void neigbours_count(const float *points, int *vector_degree, int *vector_type, const int num_points, const float *eps, const int min_pts) {{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_points) {{
            float eps2 = eps[0] * eps[0];
            float x1 = points[idx * 2];
            float y1 = points[idx * 2 + 1];
            int count = 0;
            for (int j = 0; j < num_points; j++) {{
                if (j == idx) continue; // Skip self-loop
                float x2 = points[j * 2];
                float y2 = points[j * 2 + 1];
                float dx = x2 - x1;
                float dy = y2 - y1;
                float distance = (dx * dx + dy * dy);
                if (distance <= eps2) {{
                    count++;
                }}
            }}
            vector_degree[idx] = count ; // Store the neighbor count
            if (count + 1 >= min_pts) {{
                vector_type[idx] = 1; // Core point
            }} else {{
                vector_type[idx] = -1;
            }}
        }}
    }}
    '''
    module = cp.RawModule(code=kernel_code)
    neigbours_count_kernel = module.get_function('neigbours_count')
    neigbours_count_kernel((blocks_per_grid,), (THREADS_PER_BLOCK,), (points, vector_degree, vector_type, int(num_points), eps, int(min_pts)))
    return vector_degree, vector_type


def build_adjacency_list_from_indexes(points, eps, vector_degree, adjacent_indexes):
    """
    Build the adjacency list from precomputed indexes.

    Parameters:
        points (cp.ndarray): Flattened array of points
        eps (cp.ndarray): Epsilon value
        vector_degree (cp.ndarray): Neighbor counts for each point
        adjacent_indexes (cp.ndarray): Start indexes in adjacency list

    Returns:
        cp.ndarray: Flattened adjacency list
    """
    num_points = len(points) // 2
    
    total_neighbors = int(vector_degree.sum())
    if total_neighbors == 0:
        return cp.zeros(0, dtype=cp.int32)
    
    adjacent_list = cp.zeros(total_neighbors, dtype=cp.int32)
    
    blocks_per_grid = (num_points + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    
    kernel_code = f'''
    extern "C" __global__
    void build_adjacency_list_from_indexes(const float *points, const int *vector_degree,const int *adjacent_indexes, int *adjacent_list, const int num_points, const float *eps) {{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_points) {{
            float eps2 = eps[0] * eps[0];
            float x1 = points[idx * 2];
            float y1 = points[idx * 2 + 1];
            
            int start_idx = adjacent_indexes[idx];
            int degree = vector_degree[idx];
            int count = 0;
            
            for (int j = 0; j < num_points; j++) {{
                if (j == idx) continue; // Skip self-loop
                float x2 = points[j * 2];
                float y2 = points[j * 2 + 1];
                float dx = x2 - x1;
                float dy = y2 - y1;
                float distance = (dx * dx + dy * dy);

                if (distance <= eps2) {{
                    adjacent_list[start_idx + count] = j;
                    count++;
                }} 
                if (count >= degree) {{
                    break; // Exit early if we've found all neighbors
                }}
            }}
        }}
    }}
    '''
    module = cp.RawModule(code=kernel_code)
    build_adjacency_list_kernel = module.get_function('build_adjacency_list_from_indexes')
    build_adjacency_list_kernel((blocks_per_grid,), (THREADS_PER_BLOCK,), (points, vector_degree, adjacent_indexes, adjacent_list, int(num_points), eps))
    return adjacent_list


# ---------------------------
# DBSCAN core algorithm functions
# ---------------------------
def dbscan(points, eps, min_pts,timeEpsilon):
    """
    Run the complete DBSCAN algorithm on GPU.

    Parameters:
        points (cp.ndarray): Flattened array of points
        eps (cp.ndarray): Epsilon value
        min_pts (int): Minimum points for core point
        timeEpsilon (float): Timestamp when epsilon was computed

    Returns:
        tuple: (labels: np.ndarray, cluster_count: int)
    """
    num_points = len(points) // 2
    vector_degree, vector_type, adjacent_indexes, adjacent_list = build_graph(points, eps,min_pts)
    timeGraph = time.time() # Time after graph building
    print("gpu_dbscan: TimeGraph = ", timeGraph - timeEpsilon)
    labels = cp.full(num_points, -1, dtype=cp.int32) 
    cluster_count= dbscan_core(points, labels, vector_degree, vector_type, adjacent_indexes, adjacent_list, min_pts)
    print("gpu_dbscan: TimeDBSCAN = ", time.time() - timeGraph)
    return cp.asnumpy(labels), cluster_count

def dbscan_core(points, labels, vector_degree, vector_type, adjacent_indexes, adjacent_list, min_pts):
    """
    Core DBSCAN algorithm using CUDA library via ctypes.

    Parameters:
        points (cp.ndarray): Flattened array of points
        labels (cp.ndarray): Cluster labels (modified in-place)
        vector_degree (cp.ndarray): Neighbor counts
        vector_type (cp.ndarray): Core point markers
        adjacent_indexes (cp.ndarray): Adjacency list indexes
        adjacent_list (cp.ndarray): Adjacency list
        min_pts (int): Minimum points for core point

    Returns:
        int: Number of clusters found
    """
    num_points = len(points) // 2
    adjacent_list_size = len(adjacent_list)
    
    # Convert CuPy arrays to NumPy for ctypes compatibility
    points_np = cp.asnumpy(points).astype(np.float32)
    labels_np = cp.asnumpy(labels).astype(np.int32)
    vector_degree_np = cp.asnumpy(vector_degree).astype(np.int32)
    vector_type_np = cp.asnumpy(vector_type).astype(np.int32)
    adjacent_indexes_np = cp.asnumpy(adjacent_indexes).astype(np.int32)
    adjacent_list_np = cp.asnumpy(adjacent_list).astype(np.int32)
    
    # Get pointers to NumPy arrays
    points_ptr = points_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    labels_ptr = labels_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    vector_degree_ptr = vector_degree_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    vector_type_ptr = vector_type_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    adjacent_indexes_ptr = adjacent_indexes_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    adjacent_list_ptr = adjacent_list_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    # Call CUDA function via ctypes
    cluster_id = lib.dbscan_core_cuda(
        points_ptr, labels_ptr, vector_degree_ptr, vector_type_ptr,
        adjacent_indexes_ptr, adjacent_list_ptr, min_pts, num_points, adjacent_list_size
    )
    
    # Update labels with results
    labels[:] = cp.array(labels_np)
    
    return cluster_id
    

# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    print("gpu_dbscan: Starting GPU DBSCAN clustering")

    # Start timing
    start = time.time()
    points, std_scale, min_pts = load_data()

    points_cpu = cp.asnumpy(points)
    print(f"gpu_dbscan: Number of points to cluster: {len(points_cpu)//2}")
    
    timePoints = time.time() # Time after points extraction
    print("gpu_dbscan: TimePoints = ", timePoints - start)

    eps = get_epsilon(points, min_pts, std_scale=std_scale)
    timeEpsilon = time.time() # Time after epsilon calculation
    print("gpu_dbscan: TimeEpsilon = ", timeEpsilon - timePoints)

    labels,cluster_count = dbscan(points, eps, min_pts,timeEpsilon=timeEpsilon)
    print(f"gpu_dbscan: Number of clusters found: {cluster_count}")
    
    # End timing
    end = time.time()
    print("gpu_dbscan: Final time = ", end - start)


