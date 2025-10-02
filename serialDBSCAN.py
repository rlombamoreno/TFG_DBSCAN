import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numba import njit

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
    points = []
    y_len, x_len = image.shape
    for y in range(y_len):
        for x in range(x_len):
            if image[y, x] != color_marker:
                points.append((x, y))
    return points


def dbscan(points, eps, min_pts):
    labels = np.zeros(len(points), dtype=int) - 1
    cluster_id = 0
    clusters = []
    for point_index in range(len(points)):
        if labels[point_index] == -1:
            neighbors = find_neighbors(point_index, points, eps)
            if len(neighbors) >= min_pts:
                labels[point_index] = cluster_id
                cluster = [points[point_index]]
                for neighbor_index in neighbors:
                    if labels[neighbor_index] == -1:
                        labels[neighbor_index] = cluster_id
                        cluster.append(points[neighbor_index])
                        neighbors_aux = find_neighbors(neighbor_index, points, eps)
                        if len(neighbors_aux) >= min_pts:
                            for n in neighbors_aux:
                                if n not in neighbors:
                                    neighbors.append(n)
                    if labels[neighbor_index] == 0:
                        labels[neighbor_index] = cluster_id
                        cluster.append(points[neighbor_index])
                clusters.append(cluster)
                cluster_id += 1
            else:
                labels[point_index] = 0
    return clusters, labels

def find_neighbors(point_index, points, eps):
    neighbors = []
    for i in range(len(points)):
        if i != point_index:
            distance = np.sqrt((points[point_index][0] - points[i][0]) ** 2 + (points[point_index][1] - points[i][1]) ** 2)
            if distance <= eps:
                neighbors.append(i)
    return neighbors

@njit
def compute_kn_distances(points, k):
    kn_distances = []
    for i in range(len(points)):
        eucl_dist = []
        for j in range(len(points)):
            if i != j:
                eucl_dist.append(
                    np.sqrt(
                        ((points[i][0] - points[j][0]) ** 2) +
                        ((points[i][1] - points[j][1]) ** 2)
                    )
                )
        eucl_dist.sort()
        kn_distances.append(eucl_dist[k-1])  # k-1 porque los índices empiezan en 0
    return np.array(kn_distances)
    
def get_epsilon(points, k):
    kn_distances = compute_kn_distances(points, k)
    kn_distances = np.sort(kn_distances)[::-1]  # Orden descendente
    
    # if plot:
    #     plt.figure()
    #     plt.plot(range(1, len(kn_distances)+1), kn_distances)
    #     plt.xlabel('Points sorted by k-distance')
    #     plt.ylabel(f'Distance to {k}-th nearest neighbor')
    #     plt.title('k-distance Graph')

    # Derivada discreta para encontrar el "codo"
    diffs = np.diff(kn_distances)
    elbow_index = np.argmax(diffs)
    epsilon = kn_distances[elbow_index]
    print(f"Recommended epsilon (at elbow): {epsilon}")
    return epsilon

def paint_clusters(image, clusters):
    image_colored = np.zeros((*image.shape, 3), dtype=np.uint8)
    colors = plt.cm.get_cmap('hsv', len(clusters) + 1)
    for cluster_id, cluster in enumerate(clusters):
        color = (np.array(colors(cluster_id)[:3]) * 255).astype(np.uint8)
        for point in cluster:
            image_colored[point[1], point[0]] = color
    return image_colored

if __name__ == "__main__":
    image_bw = load_image()
    print(f"Image shape: {image_bw.shape}")
    hist = compute_histogram(image_bw)
    print(hist)
    color_marker = 0
    if hist[0] < hist[1]:
        color_marker = 1
    points = get_points_in_cluster(image_bw, color_marker)
    clusters, labels = dbscan(points, eps=get_epsilon(points, k=5), min_pts=5)
    print(f"Number of clusters: {len(clusters)}")
    image_colored = paint_clusters(image_bw, clusters)
    plt.imshow(image_colored)
    plt.show()
