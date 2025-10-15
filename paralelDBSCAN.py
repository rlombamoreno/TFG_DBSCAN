import sys
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from PIL import Image
from numba import jit
import time
import os

def load_image():
    if len(sys.argv) < 2:
        print("serialDBSCAN: Must specify one image filename")
        print("example: python3 serialDBSCAN filename.jpg")
        sys.exit(1)
    image_filename = sys.argv[1]
    image_orig = Image.open(image_filename)
    return cp.array(image_orig.convert('1'), dtype=int)

def load_std_scale():
    if len(sys.argv) != 3:
        print("serialDBSCAN: Using default std_scale=1")
        return 1.00
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
    
def compute_histogram(image):
    y_len, x_len = image.shape
    color_histogram = cp.histogram(image, bins=[0, 1, 2])[0]
    return color_histogram


def get_points_in_cluster(image, color_marker):
    y_len, x_len = image.shape
    count = count_cluster_points(image, color_marker, y_len, x_len)

    # count is an integer scalar now
    if count == 0:
        return cp.empty((0,2), dtype=cp.int32)

    points_index = cp.zeros(1, dtype=cp.int32)
    # Kernel expects int (32-bit) for points
    points = cp.zeros(count * 2, dtype=cp.int32)
    
    threads_per_block = 256
    blocks_per_grid = (y_len * x_len + (threads_per_block - 1)) // threads_per_block
    
    kernel_code = r'''
    extern "C" __global__
    void get_points_in_cluster(const int *image, int *points, const int color_marker, const int y_len, const int x_len, int *points_index) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < y_len * x_len) {
            int x = idx % x_len;
            int y = idx / x_len;
            int linear_idx = y * x_len + x;
            if (image[linear_idx] != color_marker) {
                int point_idx = atomicAdd(&points_index[0], 1);
                points[point_idx * 2]     = x;
                points[point_idx * 2 + 1] = y;
            }
        }
    }
    '''
    module = cp.RawModule(code=kernel_code)
    get_points_kernel = module.get_function('get_points_in_cluster')
    get_points_kernel((blocks_per_grid,), (threads_per_block,),(image.ravel().astype(cp.int32), points, color_marker, y_len, x_len, points_index))

    # Get the actual number of points written by the kernel
    num_points = int(points_index.get()[0])
    if num_points == 0:
        return cp.empty((0,2), dtype=cp.int32)

    return points


def count_cluster_points(image, color_marker, y_len, x_len):
    count = cp.zeros(1, dtype=cp.int32)
    
    threads_per_block = 256
    blocks_per_grid = (y_len * x_len + (threads_per_block - 1)) // threads_per_block
    
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
    count_kernel((blocks_per_grid,), (threads_per_block,),(image.ravel().astype(cp.int32), count, color_marker, y_len, x_len))
    # Transfer the result to host and return as Python int
    return int(count.get()[0])

def get_epsilon(points, k,std_scale):
    kn_distances = compute_kn_distances(points, k) # k-distances
    # Heuristic: epsilon = mean + std_dev * std_scale
    epsilon = cp.mean(kn_distances) + cp.std(kn_distances) * std_scale
    print(f"Recommended epsilon: {float(epsilon)}")
    return epsilon

def compute_kn_distances(points, k):
    
    num_points = len(points) // 2
    kn_distances = cp.empty(num_points, dtype=cp.float64)

    threads_per_block = 256
    blocks_per_grid = (num_points + (threads_per_block - 1)) // threads_per_block

    kernel_code =  r'''
    extern "C" __global__
    void compute_kn_distances(const int *points, double *kn_distances, const int num_points, const int k) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_points) {
            int x1 = points[idx * 2];
            int y1 = points[idx * 2 + 1];
            double *min_dists = (double*)malloc(k * sizeof(double));
            for (int i = 0; i < k; i++) {
                min_dists[i] = 1e20; // Valor grande
            }
            
            for (int j = 0; j < num_points; j++) {
                if (j != idx) {
                    int x2 = points[j * 2];
                    int y2 = points[j * 2 + 1];
                    double dx = double(x1 - x2);
                    double dy = double(y1 - y2);
                    double dist = (dx * dx + dy * dy);
                    if (dist < min_dists[k-1]) {
                        for (int pos = 0; pos < k; pos++) {
                            if (dist < min_dists[pos]) {
                                // Desplazar y insertar
                                for (int m = k-1; m > pos; m--) {
                                    min_dists[m] = min_dists[m-1];
                                }
                                min_dists[pos] = dist;
                                break;
                            }
                        }
                    }
                }
            }
            kn_distances[idx] = sqrt(min_dists[k-1]);
            free(min_dists);
        }
    }
    '''
    module = cp.RawModule(code=kernel_code)
    compute_kn_distances_kernel = module.get_function('compute_kn_distances')
    compute_kn_distances_kernel((blocks_per_grid,), (threads_per_block,), (points, kn_distances, num_points, k))
    return kn_distances

# def dbscan(points, eps, min_pts):
#     labels = np.zeros(len(points), dtype=np.int64) - 1
#     labels, cluster_count= dbscan_core(points, eps, min_pts, labels)
#     return labels,cluster_count

# def dbscan_core(points, eps, min_pts, labels):
#     cluster_id = cp.zeros(1, dtype=cp.int32)
#     num_points = len(points) // 2
    
#     threads_per_block = 256
#     blocks_per_grid = (num_points + (threads_per_block - 1)) // threads_per_block
#     kernel_code = r'''
#     extern "C" __global__
#     void dbscan_kernel(const int *points, const int num_points, const float eps, const int min_pts, int *labels, int *cluster_id) {
#         int idx = blockIdx.x * blockDim.x + threadIdx.x;
#         if (idx < num_points) {
#             // Implement DBSCAN logic here
#         }
#     }
#     '''

#     module = cp.RawModule(code=kernel_code)
#     dbscan_kernel = module.get_function('dbscan_kernel')
#     dbscan_kernel((blocks_per_grid,), (threads_per_block,), (points, num_points, eps, min_pts, labels, cluster_id))
#     return label, int(cluster_id.get()[0])
    

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
    points_cpu = cp.asnumpy(points)
    print(f"Number of points to cluster: {len(points_cpu)//2}")
    
    timePoints = time.time() # Time after points extraction
    print("TimePoints = ", timePoints - start)
    
    eps = get_epsilon(points, k=5,std_scale=std_scale)
    timeEpsilon = time.time() # Time after epsilon calculation
    print("TimeEpsilon = ", timeEpsilon - start)
    
    # labels,cluster_count = dbscan(points, eps, min_pts=5)
    # print(f"Number of clusters found: {cluster_count}")
    
    
    
