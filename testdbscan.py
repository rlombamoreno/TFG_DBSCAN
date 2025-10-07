import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from sklearn.cluster import DBSCAN


def load_image():
    if len(sys.argv) != 2:
        print("serialDBSCAN: Must specify one image filename")
        print("example: python3 serialDBSCAN filename.jpg")
        sys.exit(1)
    image_filename = sys.argv[1]
    image_orig = Image.open(image_filename)
    return np.array(image_orig.convert('1'), dtype=int)

if __name__ == "__main__":
    start = time.time()
    image_bw = load_image()
    print(f"Image shape: {image_bw.shape}")

    # Extraer coordenadas de los píxeles con valor 1 (puedes cambiar a 0 si lo prefieres)
    points = np.column_stack(np.where(image_bw == 1))
    # Si quieres (x, y) en vez de (y, x):
    points = points[:, ::-1]

    dbscan = DBSCAN(eps=3.337191107070134, min_samples=5)
    labels = dbscan.fit_predict(points)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters found: {n_clusters}")

    # Visualización opcional de los clusters
    plt.figure(figsize=(6, 6))
    plt.imshow(image_bw, cmap='gray')
    for cluster_id in range(n_clusters):
        cluster_points = points[labels == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=2, label=f'Cluster {cluster_id}')
    plt.title('DBSCAN Clusters')
    plt.show()


