"""
gpu_dbscan.py
-------------
GPU implementation of the DBSCAN algorithm using CuPy and CUDA kernels.

Author: Rodrigo Lomba Moreno
Institution: Universidad Politécnica de Madrid  
Date: November 2025  
Version: 1.0

Description:
This script implements the DBSCAN algorithm for image segmentation using
CuPy for GPU acceleration and CUDA kernels for parallel computation.
It supports input from common image files (JPEG/PNG) and NetCDF files.

Usage:
    python3 gpu_dbscan.py <input_filename> [std_scale]

    - <input_filename> : Path to an image (.jpg, .png) or a NetCDF (.nc) file.
    - [std_scale]      : Optional float in [0, 1] used to scale the std in the
                         epsilon heuristic. If omitted, std_scale defaults to 1.0.
    - [min_pts]         : Optional integer for minimum points parameter.
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

# ---------------------------
# Data loading functions
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
    Otherwise, calculate as 2*dimension + 1.
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
    # color_marker indicates the background color;
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
    count_kernel((blocks_per_grid,), (THREADS_PER_BLOCK,),(image.ravel().astype(cp.int32), count, color_marker, y_len, x_len))
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
        k (int): Number of neighbors for k-distance
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
    return labels, cluster_count


def dbscan_core(points, labels, vector_degree, vector_type, adjacent_indexes, adjacent_list, min_pts):
    """
    Core DBSCAN algorithm using CUDA kernels for cluster expansion.

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
    cluster_id = 0
    num_points = len(points) // 2
    
    kernel_func = define_expand_cluster_kernel_gpu()

    blocks_per_grid = (num_points + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    
    border_points = cp.zeros(num_points, dtype=cp.int32)
    active_flag = cp.zeros(1, dtype=cp.int32)
    
    for idx in range(num_points):
        if int(labels[idx]) == -1 and int(vector_type[idx]) == 1:
            border_points[idx] = 1
            while True:
                active_flag[0] = 0
                kernel_func((blocks_per_grid,), (THREADS_PER_BLOCK,), (vector_degree, adjacent_indexes, adjacent_list, border_points, num_points, cluster_id, labels,min_pts,active_flag))
                if active_flag[0] == 0:
                    break
            border_points[:] = 0
            cluster_id += 1
    return  cluster_id


def define_expand_cluster_kernel_gpu():
    """
    Define and compile the CUDA kernel for cluster expansion.

    Returns:
        cupy.RawKernel: Compiled kernel function
    """
    kernel_code =  r'''
    extern "C" __global__
    void expand_cluster_kernel(const int *vector_degree, const int *adjacent_indexes, const int *adjacent_list, 
                                int *border_points, const int num_points, const int cluster_id, int *labels, const int min_pts, int *active_flag) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_points) {
            if (border_points[idx] != 0 && labels[idx] == -1) {
                labels[idx] = cluster_id;           
                int start_idx = adjacent_indexes[idx];
                int degree = vector_degree[idx];
                if(degree + 1 >= min_pts) { // Only expand if core point
                    *active_flag = 1;
                    for (int j = start_idx; j < start_idx + degree; j++) {
                        int neighbor = adjacent_list[j];
                        border_points[neighbor] = 1;
                    }
                }
            }
        }
    }
    '''
    module = cp.RawModule(code=kernel_code)
    expand_cluster_kernel_func = module.get_function('expand_cluster_kernel')
    return expand_cluster_kernel_func    


# ---------------------------
# Cluster properties computation
# ---------------------------
def compute_cluster_properties(points, labels, cluster_count):
    """
    Compute cluster properties: mass center and gyration radius.
    
    Parameters:
        points (cp.ndarray): Flattened array of points (x0, y0, x1, y1, ...)
        labels (cp.ndarray): Cluster labels for each point
        cluster_count (int): Number of clusters found
        
    Returns:
        tuple: (cluster_centers, cluster_radii, cluster_eigenvalues, cluster_sizes)
    """
    num_points = len(points) // 2
    valid_labels = labels[labels != -1]
    if len(valid_labels) == 0:
        print("gpu_dbscan: No clusters found for property computation.")
        return (cp.empty(0, dtype=cp.float32), cp.empty(0, dtype=cp.float32), 
                cp.empty(0, dtype=cp.float32), cp.empty(0, dtype=cp.int32))

    cluster_sizes_hist = cp.histogram(valid_labels, bins=cluster_count, range=(0, cluster_count))[0]
    cluster_sizes = cp.zeros(cluster_count, dtype=cp.int32)
    
    for i in range(cluster_count):
        cluster_sizes[i] = cluster_sizes_hist[i]
    
    cluster_pos = cp.zeros(cluster_count, dtype=cp.int32)
    cluster_offsets = cp.zeros(cluster_count , dtype=cp.int32)
    if cluster_count > 1:
        cluster_offsets[1:] = cp.cumsum(cluster_sizes[:-1])
    cluster_indices = cp.zeros(len(valid_labels), dtype=cp.int32)
    
    blocks_per_grid = (num_points + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    
    kernel_code = r'''
    extern "C" __global__
    void organize_cluster_points(const int *labels, const int *cluster_offsets, int *cluster_pos,
                                int *cluster_indices, int num_points) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_points) {
            int cluster_id = labels[idx];
            if (cluster_id >= 0) {
                int pos = atomicAdd(&cluster_pos[cluster_id], 1);
                cluster_indices[cluster_offsets[cluster_id] + pos] = idx;
            }
        }
    }
    '''
    module = cp.RawModule(code=kernel_code)
    kernel = module.get_function('organize_cluster_points')
    kernel((blocks_per_grid,), (THREADS_PER_BLOCK,), (labels, cluster_offsets, cluster_pos, cluster_indices, num_points))
    cluster_centers  = cp.zeros(cluster_count * 2, dtype=cp.double)
    cluster_radii = cp.zeros(cluster_count, dtype=cp.double)
    cluster_eigenvalues = cp.zeros(cluster_count * 2, dtype=cp.double)
    
    blocks_per_grid_compute = (cluster_count + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    
    kernel_code_compute = r'''
    extern "C" __global__
    void compute_cluster_properties(const float *points, const int *cluster_indices,
                                   const int *cluster_offsets, const int *cluster_sizes,
                                   double *cluster_centers, double *cluster_radii, double *cluster_eigenvalues, int cluster_count) {
        int cluster_id = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (cluster_id < cluster_count) {
            int cluster_size = cluster_sizes[cluster_id];
            
            int start_idx = cluster_offsets[cluster_id];
            
            // CDM
            double sum_x = 0.0;
            double sum_y = 0.0;
            for (int i = 0; i < cluster_size; i++) {
                int point_idx = cluster_indices[start_idx + i];
                sum_x += (double)points[point_idx * 2];
                sum_y += (double)points[point_idx * 2 + 1];
            }
            
            double center_x = sum_x / cluster_size;
            double center_y = sum_y / cluster_size;
            
            cluster_centers[cluster_id * 2] = center_x;
            cluster_centers[cluster_id * 2 + 1] = center_y;
            
            // Gyration radius
            double sum_r2 = 0.0;
            double Ixx = 0.0;
            double Iyy = 0.0;
            double Ixy = 0.0;
            for (int i = 0; i < cluster_size; i++) {
                int point_idx = cluster_indices[start_idx + i];
                double dx = (double)points[point_idx * 2] - center_x;
                double dy = (double)points[point_idx * 2 + 1] - center_y;
                
                sum_r2 += (dx * dx + dy * dy);
                
                // Inertia tensor components
                Ixx += dy * dy;
                Iyy += dx * dx;
                Ixy += dx * dy;
            }
            cluster_radii[cluster_id] = (float)sqrt(sum_r2 / cluster_size);
            Ixy = -Ixy;  
            
            double trace = Ixx + Iyy;
            double discriminant = sqrt((Ixx - Iyy) * (Ixx - Iyy) + 4.0 * Ixy * Ixy);
            double lambda1 = (trace + discriminant) / 2.0;
            double lambda2 = (trace - discriminant) / 2.0;
            cluster_eigenvalues[cluster_id * 2] = lambda1;
            cluster_eigenvalues[cluster_id * 2 + 1] = lambda2;
        }
    }
    '''
    module_compute = cp.RawModule(code=kernel_code_compute)
    kernel_compute = module_compute.get_function('compute_cluster_properties')
    kernel_compute((blocks_per_grid_compute,), (THREADS_PER_BLOCK,),
                  (points, cluster_indices, cluster_offsets, cluster_sizes,
                   cluster_centers, cluster_radii, cluster_eigenvalues, cluster_count))
    return cluster_centers, cluster_radii, cluster_eigenvalues, cluster_sizes


#---------------------------
#Save cluster properties
#---------------------------
def save_cluster_properties(cluster_centers, cluster_radii, cluster_eigenvalues, cluster_sizes, 
                           input_filename=None, std_scale=None, min_pts=None):
    """
    Save cluster properties to a file in results/GPU folder with descriptive name.
    """
    cluster_count = len(cluster_sizes)
    
    import os
    
    # Create results/GPU directory if it doesn't exist
    output_dir = "results/GPU"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    if input_filename is None:
        filename = os.path.join(output_dir, "cluster_properties.txt")
    else:
        base_name = os.path.splitext(os.path.basename(input_filename))[0]
        
        # Include parameters in filename
        param_str = ""
        if std_scale is not None and min_pts is not None:
            param_str = f"_s{std_scale}_m{min_pts}"
        elif std_scale is not None:
            param_str = f"_s{std_scale}"
        elif min_pts is not None:
            param_str = f"_m{min_pts}"
            
        filename = os.path.join(output_dir, f"cluster_properties_{base_name}{param_str}.txt")
    
    with open(filename, 'w') as f:
        # Write header with parameters
        if input_filename:
            f.write(f"# Input file: {input_filename}\n")
        if std_scale is not None:
            f.write(f"# std_scale: {std_scale}\n")
        if min_pts is not None:
            f.write(f"# min_pts: {min_pts}\n")
        f.write("# cluster_id num_particles center_x center_y gyration_radius lambda1 lambda2\n")
        
        for i in range(cluster_count):
            if cluster_sizes[i] > 0:
                center_x = cluster_centers[i * 2]
                center_y = cluster_centers[i * 2 + 1]
                radius = cluster_radii[i]
                lambda1 = cluster_eigenvalues[i * 2]
                lambda2 = cluster_eigenvalues[i * 2 + 1]
                size = cluster_sizes[i]
                
                f.write(f"{i} {size} {center_x:.6f} {center_y:.6f} {radius:.6f} {lambda1:.6f} {lambda2:.6f}\n")
    
    print(f"gpu_dbscan: Cluster properties saved to {filename}")

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

    labels, cluster_count = dbscan(points, eps, min_pts, timeEpsilon=timeEpsilon)
    print(f"gpu_dbscan: Number of clusters found: {cluster_count}")
    timeDBSCAN = time.time()
    print("gpu_dbscan: Time DBSCAN and graph construction = ", timeDBSCAN - timeEpsilon)
    
    cluster_centers, cluster_radii, cluster_eigenvalues, cluster_sizes = compute_cluster_properties(points, labels, cluster_count)
    timeProperties = time.time()
    print("gpu_dbscan: Time Properties = ", timeProperties - timeDBSCAN)
    cluster_centers_cpu = cp.asnumpy(cluster_centers)
    cluster_radii_cpu = cp.asnumpy(cluster_radii)
    cluster_eigenvalues_cpu = cp.asnumpy(cluster_eigenvalues)
    cluster_sizes_cpu = cp.asnumpy(cluster_sizes)
    
    input_filename = sys.argv[1]  # Get the input filename from command line
    save_cluster_properties(cluster_centers_cpu, cluster_radii_cpu, cluster_eigenvalues_cpu, cluster_sizes_cpu, input_filename=input_filename, std_scale=std_scale, min_pts=min_pts)
    
    # End timing
    end = time.time()
    print("gpu_dbscan: Total Time = ", end - start)
    


