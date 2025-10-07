import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numba import njit
import time
import os

def load_image():
    if len(sys.argv) < 2:
        print("serialDBSCAN: Must specify one image filename")
        print("example: python3 serialDBSCAN filename.jpg")
        sys.exit(1)
    image_filename = sys.argv[1]
    image_orig = Image.open(image_filename)
    return np.array(image_orig.convert('1'), dtype=int)

def load_std_scale():
    if len(sys.argv) != 3:
        print("serialDBSCAN: Using default std_scale=1")
        return 1
    std_scale = float(sys.argv[2])
    if std_scale < 0 or std_scale > 1:
        print("serialDBSCAN: std_scale must be between 0 and 1")
        sys.exit(1)
    return std_scale

def save_image(image_colored):
    image_filename = sys.argv[1]
    name, ext = os.path.splitext(image_filename)
    output_filename = f"{name}_clusters_CPU.png"
    plt.imsave(output_filename, image_colored)
    print(f"Clustered image saved as: {output_filename}")

@njit
def compute_histogram(image):
    y_len, x_len = image.shape
    color_histogram = np.histogram(image, bins=[0, 1, 2])[0]
    return color_histogram

@njit
def get_points_in_cluster(image, color_marker):
    y_len, x_len = image.shape
    count = count_cluster_points(image, color_marker, y_len, x_len)
    points = np.zeros((count, 2), dtype=np.int64)
    point_index = 0
    for y in range(y_len):
        for x in range(x_len):
            if image[y, x] != color_marker:
                points[point_index, 0] = x
                points[point_index, 1] = y
                point_index += 1
    return points

@njit
def count_cluster_points(image, color_marker, y_len, x_len):
    count = 0
    for y in range(y_len):
        for x in range(x_len):
            if image[y, x] != color_marker:
                count += 1
    return count

@njit
def dbscan_core(points, eps, min_pts, labels):
    cluster_id = 0
    for point_index in range(len(points)):
        if labels[point_index] == -1:
            neighbors = find_neighbors(point_index, points, eps)
            if len(neighbors)+1 >= min_pts: # +1 to include the point itself
                cluster_id += 1
                labels[point_index] = cluster_id
                i = 0
                while i < len(neighbors):
                    neighbor_index = neighbors[i]
                    if labels[neighbor_index] == -1:
                        labels[neighbor_index] = cluster_id
                        neighbors_aux = find_neighbors(neighbor_index, points, eps)
                        if len(neighbors_aux)+1 >= min_pts:
                            neighbors = append_neighbors(neighbors, neighbors_aux)
                    if labels[neighbor_index] == 0:
                        labels[neighbor_index] = cluster_id
                    i += 1
            else:
                labels[point_index] = 0
    return labels, cluster_id

@njit
def dbscan(points, eps, min_pts):
    labels = np.zeros(len(points), dtype=np.int64) - 1
    labels, cluster_count= dbscan_core(points, eps, min_pts, labels)
    return labels,cluster_count

@njit
def find_neighbors(point_index, points, eps):
    count = 0
    for i in range(len(points)):
        if i != point_index:
            dx = points[point_index][0] - points[i][0]
            dy = points[point_index][1] - points[i][1]
            distance = np.sqrt(dx * dx + dy * dy)
            if distance <= eps:
                count += 1
    neighbors = np.empty(count, dtype=np.int64)
    idx = 0
    for i in range(len(points)):
        if i != point_index:
            dx = points[point_index][0] - points[i][0]
            dy = points[point_index][1] - points[i][1]
            distance = np.sqrt(dx * dx + dy * dy)
            if distance <= eps:
                neighbors[idx] = i
                idx += 1
    return neighbors

@njit
def append_neighbors(neighbors, neighbors_aux):
    temp = np.empty(len(neighbors) + len(neighbors_aux), dtype=np.int64)
    count = 0
    for i in range(len(neighbors)):
        temp[count] = neighbors[i]
        count += 1
    for i in range(len(neighbors_aux)):
        found = False
        for j in range(len(neighbors)):  
            if neighbors_aux[i] == neighbors[j]:
                found = True
                break
        if not found:
            temp[count] = neighbors_aux[i]
            count += 1
    return temp[:count]

# Compute k-nearest neighbor distances
@njit
def compute_kn_distances(points, k):
    kn_distances = np.empty(len(points), dtype=np.float64)
    for i in range(len(points)):
        eucl_dist = np.empty(len(points)-1, dtype=np.float64)
        idx = 0
        for j in range(len(points)):
            if i != j:
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                eucl_dist[idx] = np.sqrt(dx * dx + dy * dy)
                idx += 1
        eucl_dist.sort()
        kn_distances[i] = eucl_dist[k-1]
    return kn_distances

def get_epsilon(points, k,std_scale):
    kn_distances = compute_kn_distances(points, k) # k-distances
    # Heuristic: epsilon = mean + std_dev * std_scale
    epsilon = np.mean(kn_distances) + np.std(kn_distances) * std_scale
    print(f"Recommended epsilon: {epsilon}")
    return epsilon

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
    noise_mask = labels == -1
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


if __name__ == "__main__":
    # Start timing
    start = time.time()
    
    # Load image and std_scale
    image_bw = load_image()
    std_scale=load_std_scale()
    
    # Determine color marker based on histogram
    hist = compute_histogram(image_bw)
    color_marker = 1 if hist[0] < hist[1] else 0
    
    # Extract points from the image
    points = get_points_in_cluster(image_bw, color_marker)
    
    # Compute epsilon using k-NN distances
    eps = get_epsilon(points, k=5,std_scale=std_scale)
    timeEpsilon = time.time() # Time after epsilon calculation
    print("TimeEpsilon = ", timeEpsilon - start)
    
    # Run DBSCAN
    labels,cluster_count = dbscan(points, eps, min_pts=5)
    print(f"Number of clusters found: {cluster_count}")
    
    # End timing
    end = time.time()
    print("Final time = ", end - start)
    
    # Paint clusters on the image
    image_colored = paint_clusters(image_bw, points, labels, cluster_count, color_marker)
    save_image(image_colored) # Save the colored image
    
    plt.imshow(image_colored)
    plt.show()
