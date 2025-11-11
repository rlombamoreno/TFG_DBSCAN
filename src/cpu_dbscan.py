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
# 6. Main script
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

    
    time_end = time.perf_counter()
    print("cpu_dbscan: Final time = ", time_end - time_start)
    
    
    # Visualize and save results
    input_filename = sys.argv[1]
    
    if is_image:
        # For images: paint clusters on the original image
        image_bw, color_marker = image_data
        clustered_image = save_clustered_image(
            image_bw, points, labels, cluster_count, color_marker, input_filename
        )
    else:
        # For NetCDF data: create a scatter plot
        plot_clusters(points, labels, cluster_count, input_filename)

