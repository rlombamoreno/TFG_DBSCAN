import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numba import jit
import netCDF4 as nc
import time
import os

def load_data():
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

def load_netcdf(netcdf_filename):
    ncdata = nc.Dataset(netcdf_filename)
    frame = 1
    atoms = ncdata.variables['coordinates'][:][frame]
    ncdata.close()
    
    # Convertir a array regular si es masked array
    if np.ma.isMaskedArray(atoms):
        atoms = np.ma.filled(atoms, fill_value=np.nan)
    r = atoms.transpose()
    points = points_to_array(r)
    return points

def points_to_array(r):
    x = r[0]
    y = r[1]
    points = np.zeros((len(x), 2), dtype=np.float32)
    for i in range(len(x)):
        points[i, 0] = float(x[i])
        points[i, 1] = float(y[i])
    return points

def load_image(image_filename):
    image_orig = Image.open(image_filename)
    print(f"cpu_dbscan: Image name: {image_filename} Size: {image_orig.size}")
    image_bw = np.array(image_orig.convert('1'), dtype=int)
    hist = compute_histogram(image_bw)
    hist = compute_histogram(image_bw)
    color_marker = 1 if hist[0] < hist[1] else 0
    points = get_points_in_cluster(image_bw, color_marker)
    return points

def load_std_scale():
    if len(sys.argv) != 3:
        print("cpu_dbscan: Using default std_scale=1")
        return 1.00
    std_scale = float(sys.argv[2])
    if std_scale < 0 or std_scale > 1:
        print("cpu_dbscan: std_scale must be between 0 and 1")
        sys.exit(1)
    return std_scale

def save_image(image_colored):
    image_filename = sys.argv[1]
    name, ext = os.path.splitext(image_filename)
    output_filename = f"{name}_clusters_CPU.png"
    plt.imsave(output_filename, image_colored)
    print(f"cpu_dbscan: Clustered image saved as: {output_filename}")

@jit(nopython=True)
def compute_histogram(image):
    y_len, x_len = image.shape
    color_histogram = np.histogram(image, bins=[0, 1, 2])[0]
    return color_histogram

@jit(nopython=True)
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

@jit(nopython=True)
def count_cluster_points(image, color_marker, y_len, x_len):
    count = 0
    for y in range(y_len):
        for x in range(x_len):
            if image[y, x] != color_marker:
                count += 1
    return count

@jit(nopython=True)
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

@jit(nopython=True)
def dbscan(points, eps, min_pts):
    labels = np.zeros(len(points), dtype=np.int64) - 1
    labels, cluster_count= dbscan_core(points, eps, min_pts, labels)
    return labels,cluster_count

@jit(nopython=True)
def find_neighbors(point_index, points, eps):
    count = 0
    eps2 = np.float64(eps) * np.float64(eps)
    for i in range(len(points)):
        if i != point_index:
            dx = points[point_index][0] - points[i][0]
            dy = points[point_index][1] - points[i][1]
            distance = (dx * dx + dy * dy)
            if distance <= eps2:
                count += 1
    neighbors = np.empty(count, dtype=np.int64)
    idx = 0
    for i in range(len(points)):
        if i != point_index:
            dx = points[point_index][0] - points[i][0]
            dy = points[point_index][1] - points[i][1]
            distance = (dx * dx + dy * dy)
            if distance <= eps2:
                neighbors[idx] = i
                idx += 1
    return neighbors

@jit(nopython=True)
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
@jit(nopython=True)
def compute_kn_distances(points, k):
    kn_distances = np.empty(len(points), dtype=np.float64)
    for i in range(len(points)):
        eucl_distance= np.full(k,1e20, dtype=np.float64)
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

def get_epsilon(points, k,std_scale):
    kn_distances = compute_kn_distances(points, k) # k-distances
    # Heuristic: epsilon = mean + std_dev * std_scale
    epsilon = np.mean(kn_distances) + np.std(kn_distances) * std_scale
    print(f"cpu_dbscan: Recommended epsilon: {epsilon}")
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


if __name__ == "__main__":
    print("cpu_dbscan: Starting CPU DBSCAN clustering")
    
    # Start timing
    start = time.time()
    
    # Extract data
    points, std_scale = load_data()
    print(f"cpu_dbscan: Number of points extracted: {len(points)}")
    timePoints = time.time() # Time after points extraction
    print("cpu_dbscan: TimePoints = ", timePoints - start)

    # Compute epsilon using k-NN distances
    eps = get_epsilon(points, k=5,std_scale=std_scale)
    timeEpsilon = time.time() # Time after epsilon calculation
    print("cpu_dbscan: TimeEpsilon = ", timeEpsilon - timePoints)

    # Run DBSCAN
    labels,cluster_count = dbscan(points, eps, min_pts=5)
    print(f"cpu_dbscan: Number of clusters found: {cluster_count}")

    # End timing
    end = time.time()
    print("cpu_dbscan: TimeDBSCAN = ", end - timeEpsilon)
    print("cpu_dbscan: Final time = ", end - start)

    # Paint clusters on the image
    # image_colored = paint_clusters(image_bw, points, labels, cluster_count, color_marker)
    # save_image(image_colored) # Save the colored image
    
    # plt.imshow(image_colored)
    # plt.show()
