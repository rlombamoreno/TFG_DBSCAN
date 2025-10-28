"""
cpu_dbscan.py
-------------
CPU implementation of the DBSCAN algorithm optimized with NumPy.

Author: Rodrigo Lomba  
Institution: Universidad Politécnica de Madrid  
Date: October 2025  
Version: 1.0

Description:
This script implements the DBSCAN algorithm for image segmentation using
vectorized NumPy operations and an adaptive epsilon calculation method.
It supports input from common image files (JPEG/PNG) and NetCDF files
(containing coordinate arrays).

Usage:
    python3 cpu_dbscan.py <input_filename> [std_scale]

    - <input_filename> : Path to an image (.jpg, .png) or a NetCDF (.nc) file.
    - [std_scale]      : Optional float in [0, 1] used to scale the std in the
                         epsilon heuristic. If omitted, std_scale defaults to 1.0.

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


def load_data():
    """
    Load points from an image (.jpg/.png) or NetCDF (.nc) file.

    Returns:
        tuple: (points: np.ndarray of shape (N,2), std_scale: float)
    """
    
    if len(sys.argv) < 2:
        print("cpu_dbscan: Must specify one supported file, either image or netCDF")
        print("example: python3 script.py <filename> [std_scale]")
        sys.exit(1)
    std = load_std_scale()
    filename = sys.argv[1]
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".jpg" or ext == ".png":
        return load_image(filename), std
    elif ext == ".nc":
        return load_netcdf(filename), std
    else:
        print(f"cpu_dbscan: Unsupported file extension: {ext}")
        sys.exit(1)


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
    color_marker = 1 if hist[0] < hist[1] else 0
    points = get_points_in_cluster(image_bw, color_marker)
    return points


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
    
    count_total_neighbours = 0
    for i in range(len(points)):
        count = 0
        epsilon2 = np.float64(epsilon) * np.float64(epsilon)
        for j in range(len(points)):
            if i != j:
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                distance = (dx * dx + dy * dy)
                if distance <= epsilon2:
                    count += 1
        adjacency_info[i][0] = count 
        if count + 1 >= min_pts :
            is_core_point[i] = 1   
        count_total_neighbours += count
    return count_total_neighbours


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
    time_adjacency_info = time.time()
    print("cpu_dbscan: Time graph construction = ", time_adjacency_info - time_epsilon)
    labels = np.zeros(len(points), dtype=np.int32) - 1
    cluster_count = dbscan_core(points, epsilon, min_pts, labels, is_core_point, adjacent_list, adjacency_info)
    print("cpu_dbscan: Time DBSCAN = ", time.time() - time_adjacency_info)
    return labels,cluster_count


# ---------------------------
# Cluster visualization functions
# ---------------------------
def paint_clusters(image, points, labels, cluster_count, color_marker):
    # Use color_marker to determine background and foreground
    if color_marker == 1:
        base = (1 - image) * 255 # white background
    else:
        base = image * 255 # black background

    # Create an RGB image from the base
    image_rgb = np.stack([base, base, base], axis=-1).astype(np.uint8)

    # If no clusters or no points, return the base image
    if cluster_count == 0 or len(points) == 0:
        return image_rgb

    # Separate noise and clusters
    noise_mask = labels == 0
    cluster_mask = ~noise_mask

    # Paint noise in dark red or black (optional: use black for consistency)
    if np.any(noise_mask):
        noise_pts = points[noise_mask]
        image_rgb[noise_pts[:, 1], noise_pts[:, 0]] = [0, 0, 0]  # noise in black

    # Paint clusters with colors
    if np.any(cluster_mask):
        cluster_pts = points[cluster_mask]
        cluster_labels = labels[cluster_mask]
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)

        # Generate distinct colors (excluding black and white)
        hues = np.linspace(0, 1, n_clusters + 1)[:-1]  # avoid repetition
        colors = (plt.cm.hsv(hues)[:, :3] * 255).astype(np.uint8)

        # Map labels to colors
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        # Assign colors
        for (x, y), lab in zip(cluster_pts, cluster_labels):
            image_rgb[y, x] = label_to_color[lab]
    return image_rgb


# ---------------------------
# 6. Main script
# ---------------------------
if __name__ == "__main__":
    print("cpu_dbscan: time_starting CPU DBSCAN clustering")
    
    
    time_start = time.time()
    points, std_scale = load_data()
    print(f"cpu_dbscan: Number of points extracted: {len(points)}")
    time_points_extracted = time.time()
    print("cpu_dbscan: time points extracted = ", time_points_extracted - time_start)


    epsilon = get_epsilon(points, k=MIN_POINTS,std_scale=std_scale)
    time_epsilon = time.time()
    print("cpu_dbscan: Time epsilon obtained = ", time_epsilon - time_points_extracted)

    labels,cluster_count = dbscan(points,epsilon,time_epsilon,min_pts=MIN_POINTS)
    print(f"cpu_dbscan: Number of clusters found: {cluster_count}")

    
    time_end = time.time()
    print("cpu_dbscan: Final time = ", time_end - time_start)


    # Paint clusters on the image
    # image_colored = paint_clusters(image_bw, points, labels, cluster_count, color_marker)
    # save_image(image_colored) # Save the colored image
    # plt.imshow(image_colored)
    # plt.show()
