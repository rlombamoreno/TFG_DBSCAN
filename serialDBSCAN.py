import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numba import njit
import time
import os

def load_image():
    if len(sys.argv) != 2:
        print("serialDBSCAN: Must specify one image filename")
        print("example: python3 serialDBSCAN filename.jpg")
        sys.exit(1)
    image_filename = sys.argv[1]
    image_orig = Image.open(image_filename)
    return np.array(image_orig.convert('1'), dtype=int)

def compute_histogram(image):
    y_len, x_len = image.shape
    color_histogram = np.histogram(image, bins=[0, 1, 2])[0]
    return color_histogram

@njit
def get_points_in_cluster(image, color_marker):
    y_len, x_len = image.shape
    count = 0
    for y in range(y_len):
        for x in range(x_len):
            if image[y, x] != color_marker:
                count += 1
    points = np.empty((count, 2), dtype=np.int64)
    idx = 0
    for y in range(y_len):
        for x in range(x_len):
            if image[y, x] != color_marker:
                points[idx, 0] = x
                points[idx, 1] = y
                idx += 1
    return points

@njit
def find_neighbors(point_index, points, eps):
    count = 0
    for i in range(len(points)):
        if i != point_index:
            dx = points[point_index, 0] - points[i, 0]
            dy = points[point_index, 1] - points[i, 1]
            distance = np.sqrt(dx * dx + dy * dy)
            if distance <= eps:
                count += 1
    neighbors = np.empty(count, dtype=np.int64)
    idx = 0
    for i in range(len(points)):
        if i != point_index:
            dx = points[point_index, 0] - points[i, 0]
            dy = points[point_index, 1] - points[i, 1]
            distance = np.sqrt(dx * dx + dy * dy)
            if distance <= eps:
                neighbors[idx] = i
                idx += 1
    return neighbors

@njit
def append_neighbors(neighbors, neighbors_aux):
    # Concatenate unique elements from neighbors_aux to neighbors
    for i in range(len(neighbors_aux)):
        found = False
        for j in range(len(neighbors)):
            if neighbors_aux[i] == neighbors[j]:
                found = True
                break
        if not found:
            neighbors = np.append(neighbors, neighbors_aux[i])
    return neighbors

@njit
def compute_kn_distances(points, k):
    kn_distances = np.empty(len(points), dtype=np.float64)
    for i in range(len(points)):
        eucl_dist = np.empty(len(points)-1, dtype=np.float64)
        idx = 0
        for j in range(len(points)):
            if i != j:
                dx = points[i, 0] - points[j, 0]
                dy = points[i, 1] - points[j, 1]
                eucl_dist[idx] = np.sqrt(dx * dx + dy * dy)
                idx += 1
        eucl_dist.sort()
        kn_distances[i] = eucl_dist[k-1]
    return kn_distances

def get_epsilon(points, k):
    kn_distances = compute_kn_distances(points, k)
    kn_distances = np.sort(kn_distances)[::-1]
    diffs = np.diff(kn_distances)
    elbow_index = np.argmax(diffs)
    epsilon = kn_distances[elbow_index]
    print(f"Recommended epsilon (at elbow): {epsilon}")
    return epsilon

def dbscan(points, eps, min_pts):
    labels = np.zeros(len(points), dtype=int) - 1
    cluster_id = 0
    clusters = []
    for point_index in range(len(points)):
        if labels[point_index] == -1:
            neighbors = find_neighbors(point_index, points, eps)
            if len(neighbors) >= min_pts:
                labels[point_index] = cluster_id
                cluster = [tuple(points[point_index])]
                for neighbor_index in neighbors:
                    if labels[neighbor_index] == -1:
                        labels[neighbor_index] = cluster_id
                        cluster.append(tuple(points[neighbor_index]))
                        neighbors_aux = find_neighbors(neighbor_index, points, eps)
                        if len(neighbors_aux) >= min_pts:
                            neighbors = append_neighbors(neighbors, neighbors_aux)
                    if labels[neighbor_index] == 0:
                        labels[neighbor_index] = cluster_id
                        cluster.append(tuple(points[neighbor_index]))
                clusters.append(cluster)
                cluster_id += 1
            else:
                labels[point_index] = 0
    return clusters, labels

def paint_clusters(image, clusters):
    image_colored = np.zeros((*image.shape, 3), dtype=np.uint8)
    cmap = plt.get_cmap('hsv', len(clusters) + 1)
    color_list = [(np.array(cmap(i)[:3]) * 255).astype(np.uint8) for i in range(len(clusters))]
    for cluster_id, cluster in enumerate(clusters):
        color = color_list[cluster_id]
        for point in cluster:
            image_colored[point[1], point[0]] = color
    return image_colored

if __name__ == "__main__":
    start = time.time()
    image_bw = load_image()
    print(f"Image shape: {image_bw.shape}")
    hist = compute_histogram(image_bw)
    color_marker = 1 if hist[0] < hist[1] else 0
    points = get_points_in_cluster(image_bw, color_marker)
    eps = get_epsilon(points, k=5)
    t1 = time.time()
    print("Time1 = ", t1 - start)
    clusters, labels = dbscan(points, eps, min_pts=5)
    end = time.time()
    print("Final time = ", end - start)
    image_colored = paint_clusters(image_bw, clusters)
    
    # Guardar la imagen coloreada
    image_filename = sys.argv[1]
    name, ext = os.path.splitext(image_filename)
    output_filename = f"{name}_clusters.png"
    plt.imsave(output_filename, image_colored)
    print(f"Clustered image saved as: {output_filename}")

    plt.imshow(image_colored)
    plt.show()
