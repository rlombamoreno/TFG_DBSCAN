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
    kn_distances = compute_kn_distances(points, k)
    epsilon = np.mean(kn_distances) + np.std(kn_distances) * std_scale
    print(f"Recommended epsilon: {epsilon}")
    return epsilon

def paint_clusters(image, points, labels,cluster_count):
    image_colored = np.zeros((*image.shape, 3), dtype=np.uint8)

    # labels puede contener -1 (noise). Obtenemos labels positivos ordenados.
    pos_mask = labels != -1
    pos_labels = np.unique(labels[pos_mask]) if np.any(pos_mask) else np.array([], dtype=np.int64)

    if cluster_count == 0:
        # No hay clusters, devolver imagen en RGB (blanco/negro)
        bw = (image * 255).astype(np.uint8)
        return np.stack([bw, bw, bw], axis=-1)

    cmap = plt.get_cmap('hsv', len(pos_labels))
    color_list = [(np.array(cmap(i)[:3]) * 255).astype(np.uint8) for i in range(len(pos_labels))]
    # Mapeo label -> índice en color_list
    label_to_idx = {int(label): idx for idx, label in enumerate(pos_labels)}

    for idx in range(points.shape[0]):
        x = int(points[idx, 0])
        y = int(points[idx, 1])
        lab = int(labels[idx])
        if lab == -1:
            # noise en negro (0,0,0)
            image_colored[y, x] = np.array([0, 0, 0], dtype=np.uint8)
        else:
            color = color_list[label_to_idx[lab]]
            image_colored[y, x] = color

    return image_colored


if __name__ == "__main__":
    start = time.time()
    image_bw = load_image()
    std_scale=load_std_scale()
    hist = compute_histogram(image_bw)
    color_marker = 1 if hist[0] < hist[1] else 0
    points = get_points_in_cluster(image_bw, color_marker)
    eps = get_epsilon(points, k=5,std_scale=std_scale)
    timeEpsilon = time.time()
    print("TimeEpsilon = ", timeEpsilon - start)
    labels,cluster_count = dbscan(points, eps, min_pts=5)
    print(f"Number of clusters found: {cluster_count}")
    end = time.time()
    print("Final time = ", end - start)
    image_colored = paint_clusters(image_bw, points, labels, cluster_count)
    
    # Guardar la imagen coloreada
    image_filename = sys.argv[1]
    name, ext = os.path.splitext(image_filename)
    output_filename = f"{name}_clusters.png"
    plt.imsave(output_filename, image_colored)
    print(f"Clustered image saved as: {output_filename}")
    
    plt.imshow(image_colored)
    plt.show()
