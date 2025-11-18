"""
cpu_dbscan.py
-------------
CPU implementation of the DBSCAN algorithm optimized with NumPy.

Author: Rodrigo Lomba Moreno
Institution: Universidad Politécnica de Madrid  
Date: October 2025  
Version: 1.0

Description:
This script implements the DBSCAN algorithm for image segmentation using
vectorized NumPy operations and an adaptive epsilon calculation method.
It supports input from common image files (JPEG/PNG) and NetCDF files
(containing coordinate arrays).

Usage:
    python3 cpu_dbscan.py <input_filename> [std_scale] [min_pts]

    - <input_filename> : Path to an image (.jpg, .png) or a NetCDF (.nc) file.
    - [std_scale]      : Optional float in [0, 1] used to scale the std in the
                         epsilon heuristic. If omitted, std_scale defaults to 1.0.
    - [min_pts]        : Optional integer for minimum points parameter.
                         If omitted, calculated as 2*dimension + 1.

Dependencies:
    - numpy
    - pillow
    - matplotlib
    - netCDF4
    - numba
    - time, os, sys
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numba import jit
import netCDF4 as nc
import time
import os

# ---------------------------
# Global constants
# ---------------------------
MIN_POINTS = 5
INF = 1e20


# ---------------------------
# Data loading functions
# ---------------------------
def load_std_scale():
    """
    Load the optional standard deviation scale from command line arguments.

    Returns:
        float: std_scale in range [0, 1], default is 1.0 if not provided.
    """
    
    if len(sys.argv) != 3:
        print("cpu_dbscan: Using default std_scale=1")
        return 1.00
    std_scale = float(sys.argv[2])
    if std_scale < 0 or std_scale > 1:
        print("cpu_dbscan: std_scale must be between 0 and 1")
        sys.exit(1)
    return std_scale

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
                print("cpu_dbscan: std_scale must be between 0 and 1")
                sys.exit(1)
            print(f"cpu_dbscan: Using user-provided std_scale: {std_scale}")
        except ValueError:
            print("cpu_dbscan: std_scale must be a float between 0 and 1")
            sys.exit(1)
    else:
        print("cpu_dbscan: Using default std_scale: 1.0")
    if len(sys.argv) >= 4:
        try:
            min_pts = int(sys.argv[3])
            if min_pts < 1:
                print("cpu_dbscan: min_pts must be a positive integer")
                sys.exit(1)
        except ValueError:
            print("cpu_dbscan: min_pts must be a positive integer")
            sys.exit(1)
    
    return std_scale, min_pts


def calculate_min_pts(points, user_min_pts=None):
    """
    Calculate min_pts parameter. If user provided, use that value.
    Otherwise, calculate as 2*dimension + 1.
    For image analysis, dimension is always 2.
    """
    if user_min_pts is not None:
        print(f"cpu_dbscan: Using user-provided min_pts: {user_min_pts}")
        return user_min_pts
    
    # For image analysis, dimension is always 2 (x, y coordinates)
    dimension = 2
    calculated_min_pts = 2 * dimension + 1
    print(f"cpu_dbscan: Calculated min_pts as 2 * dimension + 1 => 2 * {dimension} + 1 = {calculated_min_pts}")
    return calculated_min_pts


def load_data():
    """
    Load points from an image (.jpg/.png) or NetCDF (.nc) file.

    Returns:
        tuple: (points: np.ndarray of shape (N,2), std_scale: float)
    """
    
    if len(sys.argv) < 2:
        print("cpu_dbscan: Must specify one supported file, either image or netCDF")
        print("example: python3 script.py <filename> [std_scale] [min_pts]")
        sys.exit(1)
        
        
    std_scale, min_pts = load_parameters()
    filename = sys.argv[1]
    ext = os.path.splitext(filename)[1].lower()
    
    is_image = False
    image_data = None
    
    if ext == ".jpg" or ext == ".png":
        points,image_bw, color_marker = load_image(filename)
        is_image = True
        image_data = (image_bw, color_marker)
    elif ext == ".nc":
        points = load_netcdf(filename)
    else:
        print(f"cpu_dbscan: Unsupported file extension: {ext}")
        sys.exit(1)
    min_pts = calculate_min_pts(points, min_pts)
    return points, std_scale, min_pts, is_image, image_data


def load_image(image_filename):
    """
    Convert a binary image to a list of points corresponding to the cluster color.

    Parameters:
        image_filename (str): Path to image file

    Returns:
        np.ndarray: Array of shape (N, 2) with coordinates of cluster points
    """
    
    image_orig = Image.open(image_filename)
    print(f"cpu_dbscan: Image name: {image_filename} Size: {image_orig.size}")
    
    image_bw = np.array(image_orig.convert('1'), dtype=int)
    hist = compute_histogram(image_bw)
    
    # color_marker indicates the background color; get_points_in_cluster extracts only foreground pixels
    color_marker = 1 if hist[0] < hist[1] else 0
    
    points = get_points_in_cluster(image_bw, color_marker)
    return points, image_bw, color_marker


def load_netcdf(netcdf_filename):
    """
    Load 2D points from a NetCDF file containing coordinates.

    Parameters:
        netcdf_filename (str): Path to NetCDF file

    Returns:
        np.ndarray: Array of shape (N, 2) with coordinates
    """
    
    ncdata = nc.Dataset(netcdf_filename)
    frame = 1
    atoms = ncdata.variables['coordinates'][:][frame]
    ncdata.close()
    
    if np.ma.isMaskedArray(atoms):
        atoms = np.ma.filled(atoms, fill_value=np.nan)
    
    r = atoms.transpose()
    points = points_to_array(r)
    return points


@jit(nopython=True)
def points_to_array(r):
    """
    Convert a 2xN coordinate array to Nx2 float32 array.

    Parameters:
        r (np.ndarray): 2xN array

    Returns:
        np.ndarray: Nx2 array of points
    """
    
    x = r[0]
    y = r[1]
    points = np.zeros((len(x), 2), dtype=np.float32)
    for i in range(len(x)):
        points[i, 0] = float(x[i])
        points[i, 1] = float(y[i])
    return points


# ---------------------------
# Image utility functions
# ---------------------------
@jit(nopython=True)
def compute_histogram(image):
    """Return the histogram of a binary image as a 2-element array [0_pixels, 1_pixels]."""
    
    y_len, x_len = image.shape
    color_histogram = np.histogram(image, bins=[0, 1, 2])[0]
    return color_histogram


@jit(nopython=True)
def get_points_in_cluster(image, color_marker):
    """
    Extracts the coordinates of all pixels that are not the background.

    Parameters:
        image (2D np.array): Binary image.
        color_marker (int): Value representing the background color.

    Returns:
        np.array: Array of shape (num_points, 2) containing the coordinates 
                  of all foreground pixels.
    """
    
    y_len, x_len = image.shape
    points_count = 0
    for y in range(y_len):
        for x in range(x_len):
            if image[y, x] != color_marker:
                points_count += 1
    
    points = np.zeros((points_count, 2), dtype=np.float32)
    point_index = 0
    for y in range(y_len):
        for x in range(x_len):
            if image[y, x] != color_marker:
                points[point_index, 0] = x
                points[point_index, 1] = y
                point_index += 1
    return points


def save_image(image_colored):
    image_filename = sys.argv[1]
    name, ext = os.path.splitext(image_filename)
    output_filename = f"{name}_clusters_CPU.png"
    plt.imsave(output_filename, image_colored)
    print(f"cpu_dbscan: Clustered image saved as: {output_filename}")


# ---------------------------
# DBSCAN algorithm functions
# ---------------------------
@jit(nopython=True)
def compute_kn_distances(points, k):
    """
    Compute the k-nearest distances for each point.

    Parameters:
        points (np.ndarray): Nx2 array of points
        k (int): Number of neighbors

    Returns:
        np.ndarray: Array of k-distances for each point
    """
    
    kn_distances = np.empty(len(points), dtype=np.float64)
    for i in range(len(points)):
        # Maintain a sorted array of the k smallest distances for the current point
        eucl_distance= np.full(k,INF, dtype=np.float64)
        for j in range(len(points)):
            if i != j:
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                dist = (dx * dx + dy * dy)
                if(dist < eucl_distance[k-1]):
                    for pos in range(k):
                        if(dist < eucl_distance[pos]):
                            for m in range(k-1, pos, -1):
                                eucl_distance[m] = eucl_distance[m-1]
                            eucl_distance[pos] = dist
                            break
                
        kn_distances[i] = np.sqrt(eucl_distance[k-1])
    return kn_distances


def get_epsilon(points, k, std_scale):
    """
    Compute the adaptive epsilon using k-distances and standard deviation.

    Parameters:
        points (np.ndarray): Nx2 array of points
        k (int): Number of neighbors for k-distance
        std_scale (float): Scaling factor for standard deviation

    Returns:
        float: Recommended epsilon
    """
    
    kn_distances = compute_kn_distances(points, k)
    epsilon = np.mean(kn_distances) + np.std(kn_distances) * std_scale
    print(f"cpu_dbscan: Recommended epsilon: {epsilon}")
    return epsilon


@jit(nopython=True)
def neighbor_count(points, epsilon, is_core_point, adjacency_info, min_pts):
    """
    Count neighbors within epsilon for each point and mark core points.

    Parameters:
        points (np.ndarray): Nx2 array of points
        epsilon (float): Distance threshold
        is_core_point (np.ndarray): Array marking core points (1 if core)
        adjacency_info (np.ndarray): Array to store neighbor counts and indices
        min_pts (int): Minimum points to consider a core point

    Returns:
        int: Total number of neighbors across all points
    """
    
    count_total_neighbors = 0
    for i in range(len(points)):
        count = 0
        # squared_epsilon is the squared epsilon to avoid computing square roots
        squared_epsilon = np.float64(epsilon) * np.float64(epsilon)
        for j in range(len(points)):
            if i != j:
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                distance = (dx * dx + dy * dy)
                if distance <= squared_epsilon:
                    count += 1
        adjacency_info[i][0] = count 
        # mark the point as core if it has enough neighbors
        if count + 1 >= min_pts:
            is_core_point[i] = 1
        count_total_neighbors += count
    return count_total_neighbors


@jit(nopython=True)
def build_adjacency_info(points, epsilon, adjacency_info,min_pts):
    """
    Build adjacency information for DBSCAN (core points and neighbor list).

    Parameters:
        points (np.ndarray): Nx2 array of points
        epsilon (float): Distance threshold
        adjacency_info (np.ndarray): Preallocated array for neighbor info
        min_pts (int): Minimum points for core point

    Returns:
        tuple:
            is_core_point (np.ndarray): Array indicating core points
            adjacent_list (np.ndarray): Flattened adjacency list of neighbor indices
    """
    
    is_core_point = np.zeros(len(points), dtype=np.int32)-1
    count_total_neigbours = neighbor_count(points, epsilon, is_core_point, adjacency_info,min_pts)
    # compute the starting index in the flattened neighbor array for each point
    adjacency_info[1:,1] = np.cumsum(adjacency_info[:-1,0])
    adjacent_list = np.zeros(count_total_neigbours, dtype=np.int32)
    index = 0
    for i in range(len(points)):
        epsilon2 = np.float64(epsilon) * np.float64(epsilon)
        for j in range(len(points)):
            if i != j:
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                distance = (dx * dx + dy * dy)
                if distance <= epsilon2:
                    # fill the flattened neighbor list with indices of neighbors for all points
                    adjacent_list[index] = j
                    index += 1
    return is_core_point, adjacent_list


@jit(nopython=True)
def dbscan_core(points, epsilon, min_pts, labels, is_core_point, adjacent_list, adjacency_info):
    """
    Main DBSCAN algorithm: expand clusters starting from core points.

    Parameters:
        points (np.ndarray): Nx2 array of points
        epsilon (float): Distance threshold
        min_pts (int): Minimum points to be a core point
        labels (np.ndarray): Cluster labels for points (modified in-place)
        is_core_point (np.ndarray): Array marking core points
        adjacent_list (np.ndarray): Flattened neighbor list
        adjacency_info (np.ndarray): Neighbor counts and start indices

    Returns:
        int: Total number of clusters found
    """
    
    cluster_id = 0
    border_points = np.zeros(len(points), dtype=np.int32)
    active_flag = 0;
    for point_index in range(len(points)):
        if labels[point_index] == -1 and is_core_point[point_index] == 1:
            # Expand the cluster starting from a core point; border_points marks the active frontier
            border_points[point_index] = 1
            while True:
                active_flag = 0
                for i in range(len(points)):
                    if border_points[i] == 1 and labels[i] == -1:
                        labels[i] = cluster_id
                        neighbor_start = adjacency_info[i][1]
                        degree = adjacency_info[i][0]
                        neighbors_time_end = neighbor_start + degree
                        if is_core_point[i] == 1:
                            # extract the neighbors of the current point from the flattened adjacency list
                            neighbors = adjacent_list[neighbor_start:neighbors_time_end]
                            active_flag = 1
                            for j in range(len(neighbors)):
                                neighbor_index = neighbors[j]
                                border_points[neighbor_index] = 1
                if active_flag == 0:
                    break
            border_points[:] = 0
            cluster_id += 1
    return cluster_id


def dbscan(points, epsilon,time_epsilon, min_pts):
    """
    Run DBSCAN with adjacency information and measure time.

    Parameters:
        points (np.ndarray): Nx2 array of points
        epsilon (float): Distance threshold
        time_epsilon (float): Timestamp when epsilon was computed
        min_pts (int): Minimum points to be a core point

    Returns:
        tuple: (labels: np.ndarray of cluster labels, cluster_count: int)
    """
    
    adjacency_info = np.zeros((len(points), 2), dtype=np.int32)
    is_core_point, adjacent_list = build_adjacency_info(points, epsilon, adjacency_info,min_pts)
    time_adjacency_info = time.perf_counter()
    print("cpu_dbscan: Time graph construction = ", time_adjacency_info - time_epsilon)
    labels = np.zeros(len(points), dtype=np.int32) - 1
    cluster_count = dbscan_core(points, epsilon, min_pts, labels, is_core_point, adjacent_list, adjacency_info)
    print("cpu_dbscan: Time DBSCAN = ", time.perf_counter() - time_adjacency_info)
    return labels,cluster_count


# ---------------------------
# Cluster properties computation
# ---------------------------
def compute_cluster_properties(points, labels, cluster_count):
    """
    Compute cluster properties: mass center and gyration radius.
    
    Parameters:
        points (np.ndarray): Array of points (N, 2) or flattened array
        labels (np.ndarray): Cluster labels for each point
        cluster_count (int): Number of clusters found
        
    Returns:
        tuple: (cluster_centers, cluster_radii, cluster_eigenvalues, cluster_sizes, cluster_eigenvalues_relation)
    """
      # Ensure points is in (N, 2) shape
    if points.ndim == 1:
        num_points = len(points) // 2
        points_2d = points.reshape(-1, 2)
    else:
        num_points = len(points)
        points_2d = points
    
    # Filter out noise points (label = -1)
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    
    if len(valid_labels) == 0:
        print("cpu_dbscan: No clusters found for property computation.")
        return (np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), 
                np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32))

    # Calculate cluster sizes using the same method as CPU
    cluster_sizes = np.zeros(cluster_count, dtype=np.int32)
    for i in range(cluster_count):
        cluster_sizes[i] = np.sum(valid_labels == i)
    
    # Organize cluster indices using the same graph structure as CPU DBSCAN
    cluster_indices = []
    cluster_offsets = np.zeros(cluster_count, dtype=np.int32)
    
    # Build cluster offsets (similar to GPU version but in series)
    if cluster_count > 0:
        cluster_offsets[0] = 0
        for i in range(1, cluster_count):
            cluster_offsets[i] = cluster_offsets[i-1] + cluster_sizes[i-1]
    
    # Initialize cluster indices array
    total_cluster_points = np.sum(cluster_sizes)
    cluster_indices_arr = np.zeros(total_cluster_points, dtype=np.int32)
    cluster_pos = np.zeros(cluster_count, dtype=np.int32)
    
    # Organize points by cluster using the valid points only
    for point_idx in range(num_points):
        cluster_id = labels[point_idx]
        if cluster_id >= 0:  # Skip noise points
            pos = cluster_pos[cluster_id]
            cluster_indices_arr[cluster_offsets[cluster_id] + pos] = point_idx
            cluster_pos[cluster_id] += 1
    
    # Initialize output arrays
    cluster_centers = np.zeros(cluster_count * 2, dtype=np.float32)
    cluster_radii = np.zeros(cluster_count, dtype=np.float32)
    cluster_eigenvalues = np.zeros(cluster_count * 2, dtype=np.float32)
    cluster_eigenvalues_relation = np.zeros(cluster_count, dtype=np.float32)
    
    # Compute properties for each cluster using the graph structure
    for cluster_id in range(cluster_count):
        cluster_size = cluster_sizes[cluster_id]
        
        if cluster_size == 0:
            continue
            
        start_idx = cluster_offsets[cluster_id]
        
        # Compute center of mass
        sum_x = 0.0
        sum_y = 0.0
        
        for i in range(cluster_size):
            point_idx = cluster_indices_arr[start_idx + i]
            sum_x += points_2d[point_idx, 0]
            sum_y += points_2d[point_idx, 1]
        
        center_x = sum_x / cluster_size
        center_y = sum_y / cluster_size
        
        cluster_centers[cluster_id * 2] = center_x
        cluster_centers[cluster_id * 2 + 1] = center_y
        
        # Compute gyration radius and inertia tensor
        sum_r2 = 0.0
        Ixx = 0.0
        Iyy = 0.0
        Ixy = 0.0
        
        for i in range(cluster_size):
            point_idx = cluster_indices_arr[start_idx + i]
            dx = points_2d[point_idx, 0] - center_x
            dy = points_2d[point_idx, 1] - center_y
            
            sum_r2 += dx * dx + dy * dy
            
            # Inertia tensor components
            Ixx += dy * dy  # Ixx = Σ y²
            Iyy += dx * dx  # Iyy = Σ x²
            Ixy += dx * dy  # Ixy = -Σ xy
        
        cluster_radii[cluster_id] = np.sqrt(sum_r2 / cluster_size)
        
        # Adjust Ixy sign for proper inertia tensor
        Ixy = -Ixy
        
        # Compute eigenvalues of inertia tensor
        trace = Ixx + Iyy
        discriminant = np.sqrt((Ixx - Iyy) * (Ixx - Iyy) + 4.0 * Ixy * Ixy)
        
        lambda1 = (trace + discriminant) / 2.0
        lambda2 = (trace - discriminant) / 2.0
        
        cluster_eigenvalues[cluster_id * 2] = lambda1
        cluster_eigenvalues[cluster_id * 2 + 1] = lambda2
        
        # Compute relation between eigenvalues
        lambda_max = max(lambda1, lambda2)
        lambda_min = min(lambda1, lambda2)
        
        if lambda_min > 1e-10:  # Avoid division by zero
            cluster_eigenvalues_relation[cluster_id] = lambda_max / lambda_min
        else:
            cluster_eigenvalues_relation[cluster_id] = 0.0
    
    return cluster_centers, cluster_radii, cluster_eigenvalues, cluster_sizes, cluster_eigenvalues_relation

#---------------------------
# Save cluster properties
#---------------------------
def save_cluster_properties(cluster_centers, cluster_radii, cluster_eigenvalues, cluster_eigenvalues_relation, cluster_sizes, 
                           input_filename=None, std_scale=None, min_pts=None):
    """
    Save cluster properties to a file in results/CPU folder with descriptive name.
    """
    cluster_count = len(cluster_sizes)
    
    import os
    
    # Create results/CPU directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(parent_dir, "results", "CPU")
    os.makedirs(results_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    
    # Include parameters in filename
    param_str = ""
    if std_scale is not None and min_pts is not None:
        param_str = f"_s{std_scale}_m{min_pts}"
    elif std_scale is not None:
        param_str = f"_s{std_scale}"
    elif min_pts is not None:
        param_str = f"_m{min_pts}"
        
    filename = os.path.join(results_dir, f"cluster_properties_{base_name}{param_str}.txt")

    with open(filename, 'w') as f:
        # Write header with parameters
        if input_filename:
            f.write(f"# Input file: {input_filename}\n")
        if std_scale is not None:
            f.write(f"# std_scale: {std_scale}\n")
        if min_pts is not None:
            f.write(f"# min_pts: {min_pts}\n")
        f.write("# cluster_id num_points center_x center_y gyration_radius lambda1 lambda2 relation_between_lambdas\n")
        
        for i in range(cluster_count):
            if cluster_sizes[i] > 0:
                center_x = cluster_centers[i * 2]
                center_y = cluster_centers[i * 2 + 1]
                radius = cluster_radii[i]
                lambda1 = cluster_eigenvalues[i * 2]
                lambda2 = cluster_eigenvalues[i * 2 + 1]
                size = cluster_sizes[i]
                relation = cluster_eigenvalues_relation[i]
                
                f.write(f"{i} {size} {center_x:.6f} {center_y:.6f} {radius:.6f} {lambda1:.6f} {lambda2:.6f} {relation:.6f}\n")
    
    print(f"gpu_dbscan: Cluster properties saved to {filename}")


# ---------------------------
# Histogram functions
# ---------------------------
def create_cluster_histograms(cluster_sizes, cluster_radii, cluster_eigenvalues_relation, 
                             input_filename=None, std_scale=None, min_pts=None):
    """
    Create three histograms: cluster sizes, gyration radii, and eigenvalue relations.
    
    Parameters:
        cluster_sizes (np.ndarray): Array of cluster sizes
        cluster_radii (np.ndarray): Array of gyration radii
        cluster_eigenvalues_relation (np.ndarray): Array of eigenvalue relations (lambda_max/lambda_min)
        input_filename (str): Original input filename for naming
        std_scale (float): std_scale parameter for filename
        min_pts (int): min_pts parameter for filename
    """
    # Filter out zero-sized clusters
    valid_mask = cluster_sizes > 0
    sizes = cluster_sizes[valid_mask]
    radii = cluster_radii[valid_mask]
    relations = cluster_eigenvalues_relation[valid_mask]
    
    if len(sizes) == 0:
        print("cpu_dbscan: No valid clusters for histogram creation.")
        return
    
    # Create results/CPU directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(parent_dir, "results", "CPU")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate base filename
    base_name = os.path.splitext(os.path.basename(input_filename))[0] if input_filename else "clusters"
    param_str = ""
    if std_scale is not None and min_pts is not None:
        param_str = f"_s{std_scale}_m{min_pts}"
    elif std_scale is not None:
        param_str = f"_s{std_scale}"
    elif min_pts is not None:
        param_str = f"_m{min_pts}"
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Histogram 1: Cluster sizes
    axes[0].hist(sizes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Tamaño del Cluster (número de puntos)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title('Distribución de Tamaños de Clusters')
    axes[0].grid(True, alpha=0.3)
    # Add statistics
    axes[0].axvline(np.mean(sizes), color='red', linestyle='--', label=f'Media: {np.mean(sizes):.1f}')
    axes[0].axvline(np.median(sizes), color='green', linestyle='--', label=f'Mediana: {np.median(sizes):.1f}')
    axes[0].legend()
    
    # Histogram 2: Gyration radii
    axes[1].hist(radii, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('Radio de Giro')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Distribución de Radios de Giro')
    axes[1].grid(True, alpha=0.3)
    # Add statistics
    axes[1].axvline(np.mean(radii), color='red', linestyle='--', label=f'Media: {np.mean(radii):.2f}')
    axes[1].axvline(np.median(radii), color='green', linestyle='--', label=f'Mediana: {np.median(radii):.2f}')
    axes[1].legend()
    
    # Histogram 3: Eigenvalue relations (shape anisotropy)
    # Filter out infinite values and very large values
    valid_relations = relations[np.isfinite(relations)]
    valid_relations = valid_relations[valid_relations < np.percentile(valid_relations, 95)]  # Remove outliers
    
    axes[2].hist(valid_relations, bins=20, alpha=0.7, color='salmon', edgecolor='black')
    axes[2].set_xlabel(r'Relación $\lambda_{\max} / \lambda_{\min}$')
    axes[2].set_ylabel('Frecuencia')
    axes[2].set_title('Distribución Relacion de Autovalores del Tensor de Inercia')
    axes[2].grid(True, alpha=0.3)
    # Add statistics and interpretation
    mean_rel = np.mean(valid_relations)
    axes[2].axvline(mean_rel, color='red', linestyle='--', label=f'Media: {mean_rel:.2f}')
    axes[2].axvline(1.0, color='blue', linestyle='-', alpha=0.5, label='Perfectamente circular')
    
    # Add shape interpretation
    if mean_rel < 1.5:
        shape_text = "Formas mayormente circulares"
    elif mean_rel < 3.0:
        shape_text = "Mezcla de formas"
    else:
        shape_text = "Formas mayormente alargadas"
    
    axes[2].text(0.05, 0.95, shape_text, transform=axes[2].transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                verticalalignment='top')
    axes[2].legend()
    
    # Adjust layout and save
    plt.tight_layout()
    histogram_filename = os.path.join(results_dir, f"histograms_{base_name}{param_str}.png")
    plt.savefig(histogram_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"cpu_dbscan: Histograms saved to {histogram_filename}")
    
    # Also save histogram data as text file
    data_filename = os.path.join(results_dir, f"histogram_data_{base_name}{param_str}.txt")
    with open(data_filename, 'w') as f:
        f.write("# Cluster Size Statistics\n")
        f.write(f"# Total clusters: {len(sizes)}\n")
        f.write(f"# Mean size: {np.mean(sizes):.2f}\n")
        f.write(f"# Median size: {np.median(sizes):.2f}\n")
        f.write(f"# Std size: {np.std(sizes):.2f}\n")
        f.write(f"# Min size: {np.min(sizes)}\n")
        f.write(f"# Max size: {np.max(sizes)}\n\n")
        
        f.write("# Gyration Radius Statistics\n")
        f.write(f"# Mean radius: {np.mean(radii):.4f}\n")
        f.write(f"# Median radius: {np.median(radii):.4f}\n")
        f.write(f"# Std radius: {np.std(radii):.4f}\n")
        f.write(f"# Min radius: {np.min(radii):.4f}\n")
        f.write(f"# Max radius: {np.max(radii):.4f}\n\n")
        
        f.write("# Shape Anisotropy Statistics\n")
        f.write(f"# Mean relation: {np.mean(valid_relations):.4f}\n")
        f.write(f"# Median relation: {np.median(valid_relations):.4f}\n")
        f.write(f"# Std relation: {np.std(valid_relations):.4f}\n")
        f.write(f"# Interpretation: {shape_text}\n")
    
    print(f"cpu_dbscan: Histogram data saved to {data_filename}")


# ---------------------------
# Visualization functions
# ---------------------------
def paint_clusters(image, points, labels, cluster_count, color_marker):
    """
    Paint clusters on the original image with different colors.
    
    Parameters:
        image (np.ndarray): Original binary image
        points (np.ndarray): Array of points used for clustering
        labels (np.ndarray): Cluster labels (-1 = noise, >=0 = cluster ID)
        cluster_count (int): Number of clusters found
        color_marker (int): Background color marker
        
    Returns:
        np.ndarray: Colored RGB image with clusters
    """
    # Use color_marker to determine background and foreground
    if color_marker == 1:
        base = (1 - image) * 255  # white background
    else:
        base = image * 255  # black background

    # Create an RGB image from the base
    image_rgb = np.stack([base, base, base], axis=-1).astype(np.uint8)

    # If no clusters or no points, return the base image
    if cluster_count == 0 or len(points) == 0:
        return image_rgb

    # Separate noise and clusters
    noise_mask = labels == -1
    cluster_mask = labels >= 0

    # Paint noise in red
    if np.any(noise_mask):
        noise_pts = points[noise_mask]
        if len(noise_pts) > 0:
            y_coords = np.clip(noise_pts[:, 1].astype(int), 0, image_rgb.shape[0] - 1)
            x_coords = np.clip(noise_pts[:, 0].astype(int), 0, image_rgb.shape[1] - 1)
            image_rgb[y_coords, x_coords] = [255, 0, 0]  # noise in red

    # Paint clusters with colors
    if np.any(cluster_mask):
        cluster_pts = points[cluster_mask]
        cluster_labels = labels[cluster_mask]
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)

        # Generate distinct colors (excluding black, white, and red)
        if n_clusters > 0:
            hues = np.linspace(0, 1, n_clusters + 1)[:-1]  # avoid repetition
            colors = (plt.cm.hsv(hues)[:, :3] * 255).astype(np.uint8)

            # Map labels to colors
            for i, label in enumerate(unique_labels):
                mask_for_label = cluster_labels == label
                colored_pts = cluster_pts[mask_for_label]
                if len(colored_pts) > 0:
                    y_coords = np.clip(colored_pts[:, 1].astype(int), 0, image_rgb.shape[0] - 1)
                    x_coords = np.clip(colored_pts[:, 0].astype(int), 0, image_rgb.shape[1] - 1)
                    image_rgb[y_coords, x_coords] = colors[i]
    
    return image_rgb


def save_clustered_image(image, points, labels, cluster_count, color_marker, input_filename):
    """
    Save clustered image to Results/CPU directory with descriptive filename.
    
    Parameters:
        image (np.ndarray): Original binary image
        points (np.ndarray): Array of points used for clustering
        labels (np.ndarray): Cluster labels
        cluster_count (int): Number of clusters found
        color_marker (int): Background color marker
        input_filename (str): Original input filename
    """
    # Create results/CPU directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(parent_dir, "results", "CPU")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    output_filename = f"{results_dir}/{base_name}_clusters_CPU.png"
    
    # Paint the clusters
    image_colored = paint_clusters(image, points, labels, cluster_count, color_marker)
    
    # Save the image
    plt.imsave(output_filename, image_colored)
    print(f"cpu_dbscan: Clustered image saved as: {output_filename}")
    
    return image_colored


def plot_clusters(points, labels, cluster_count, input_filename):
    """
    Create and save a scatter plot of the clusters for non-image data.
    
    Parameters:
        points (np.ndarray): Array of points
        labels (np.ndarray): Cluster labels
        cluster_count (int): Number of clusters found
        input_filename (str): Original input filename
    """
    # Create results/CPU directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(parent_dir, "results", "CPU")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    output_filename = f"{results_dir}/{base_name}_clusters_CPU.png"
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Separate noise and clusters
    noise_mask = labels == -1
    cluster_mask = labels >= 0
    
    # Plot noise points in red
    if np.any(noise_mask):
        noise_points = points[noise_mask]
        plt.scatter(noise_points[:, 0], noise_points[:, 1], 
                   c='red', marker='x', s=10, alpha=0.6, label='Noise')
    
    # Plot clusters with different colors
    if np.any(cluster_mask):
        cluster_points = points[cluster_mask]
        cluster_labels = labels[cluster_mask]
        unique_labels = np.unique(cluster_labels)
        
        # Generate colors for clusters
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            cluster_data = cluster_points[mask]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                       c=[colors[i]], marker='o', s=20, alpha=0.7,
                       label=f'Cluster {label}')
    
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title(f'DBSCAN Clustering - {cluster_count} clusters found')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"cpu_dbscan: Cluster plot saved as: {output_filename}")


# ---------------------------
# Main script
# ---------------------------
if __name__ == "__main__":
    print("cpu_dbscan: time_starting CPU DBSCAN clustering")
    
    time_start = time.perf_counter()
    
    points, std_scale, min_pts, is_image, image_data = load_data()
    print(f"cpu_dbscan: Number of points extracted: {len(points)}")
    time_points_extracted = time.perf_counter()
    print("cpu_dbscan: time points extracted = ", time_points_extracted - time_start)


    epsilon = get_epsilon(points, min_pts,std_scale=std_scale)
    time_epsilon = time.perf_counter()
    print("cpu_dbscan: Time epsilon obtained = ", time_epsilon - time_points_extracted)

    labels,cluster_count = dbscan(points,epsilon,time_epsilon,min_pts)
    print(f"cpu_dbscan: Number of clusters found: {cluster_count}")
    timeDBSCAN = time.perf_counter()
    print("cpu_dbscan: Time DBSCAN and graph construction = ", timeDBSCAN - time_epsilon)
    
    cluster_centers, cluster_radii, cluster_eigenvalues, cluster_sizes, cluster_eigenvalues_relation = compute_cluster_properties(points, labels, cluster_count)
    timeProperties = time.perf_counter()
    print("cpu_dbscan: Time Properties = ", timeProperties - timeDBSCAN)
    
    input_filename = sys.argv[1]  # Get the input filename from command line
    save_cluster_properties(cluster_centers, cluster_radii, cluster_eigenvalues, cluster_eigenvalues_relation, cluster_sizes, input_filename=input_filename, std_scale=std_scale, min_pts=min_pts)
    
    create_cluster_histograms(cluster_sizes, cluster_radii, cluster_eigenvalues_relation, input_filename=input_filename, std_scale=std_scale, min_pts=min_pts)
    
    time_end = time.perf_counter()
    print("cpu_dbscan: Total time = ", time_end - time_start)
    
    
    if is_image:
        # For images: paint clusters on the original image
        image_bw, color_marker = image_data
        clustered_image = save_clustered_image(
            image_bw, points, labels, cluster_count, color_marker, input_filename
        )
    else:
        # For NetCDF data: create a scatter plot
        plot_clusters(points, labels, cluster_count, input_filename)

