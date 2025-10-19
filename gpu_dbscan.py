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
        print("gpu_dbscan: Must specify one image filename")
        print("example: python3 gpu_dbscan filename.jpg")
        sys.exit(1)
    image_filename = sys.argv[1]
    image_orig = Image.open(image_filename)
    print(f"gpu_dbscan: Image name: {image_filename} Size: {image_orig.size}")
    return cp.array(image_orig.convert('1'), dtype=int)

def load_std_scale():
    if len(sys.argv) != 3:
        print("gpu_dbscan: Using default std_scale=1")
        return 1.00
    std_scale = float(sys.argv[2])
    if std_scale < 0 or std_scale > 1:
        print("gpu_dbscan: std_scale must be between 0 and 1")
        sys.exit(1)
    return std_scale

def save_image(image_colored):
    image_filename = sys.argv[1]
    name, ext = os.path.splitext(image_filename)
    output_filename = f"{name}_clusters_GPU.png"
    plt.imsave(output_filename, image_colored)
    print(f"gpu_dbscan: Clustered image saved as: {output_filename}")
    
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
    print(f"gpu_dbscan: Recommended epsilon: {float(epsilon)}")
    return epsilon

def compute_kn_distances(points, k):
    
    num_points = len(points) // 2
    kn_distances = cp.empty(num_points, dtype=cp.float64)

    threads_per_block = 256
    blocks_per_grid = (num_points + (threads_per_block - 1)) // threads_per_block
    
    MAX_K = k
    kernel_code =  f'''
    #define MAX_K {MAX_K} // Define MAX_K based on k, needed for array size
    
    extern "C" __global__
    void compute_kn_distances(const int *points, double *kn_distances, const int num_points, const int k) {{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_points) {{
            int x1 = points[idx * 2];
            int y1 = points[idx * 2 + 1];
            double min_dists[MAX_K];
            for (int i = 0; i < k; i++) {{
                min_dists[i] = 1e20; // Valor grande
            }}

            for (int j = 0; j < num_points; j++) {{
                if (j != idx) {{
                    int x2 = points[j * 2];
                    int y2 = points[j * 2 + 1];
                    double dx = double(x1 - x2);
                    double dy = double(y1 - y2);
                    double dist = (dx * dx + dy * dy);
                    if (dist < min_dists[k-1]) {{
                        for (int pos = 0; pos < k; pos++) {{
                            if (dist < min_dists[pos]) {{
                                // Desplazar y insertar
                                for (int m = k-1; m > pos; m--) {{
                                    min_dists[m] = min_dists[m-1];
                                }}
                                min_dists[pos] = dist;
                                break;
                            }}
                        }}
                    }}
                }}
            }}
            kn_distances[idx] = sqrt(min_dists[k-1]);
        }}
    }}
    '''
    module = cp.RawModule(code=kernel_code)
    compute_kn_distances_kernel = module.get_function('compute_kn_distances')
    compute_kn_distances_kernel((blocks_per_grid,), (threads_per_block,), (points, kn_distances, num_points, k))
    return kn_distances

def dbscan(points, eps, min_pts):
    num_points = len(points) // 2
    vector_degree, vector_type, adjacent_indexes, adjacent_list = build_graph(points, eps,min_pts)
    labels = cp.full(num_points, -1, dtype=cp.int32) 
    cluster_count= dbscan_core(points, labels, vector_degree, vector_type, adjacent_indexes, adjacent_list, min_pts)
    return cp.asnumpy(labels), cluster_count

def dbscan_core(points, labels, vector_degree, vector_type, adjacent_indexes, adjacent_list, min_pts):
    cluster_id = 0
    num_points = len(points) // 2
    
    for i in range(num_points):
        if vector_type[i] == 1 and labels[i] == -1: # Not visited and is core
            expand_cluster_gpu(points, i, cluster_id, labels, vector_degree, adjacent_indexes, adjacent_list, min_pts)
            cluster_id += 1
    return  cluster_id



def expand_cluster_gpu(points, point_index, cluster_id, labels, vector_degree, adjacent_indexes, adjacent_list, min_pts):
    num_points = len(points) // 2
    
    current_border_points = cp.zeros(num_points, dtype=cp.int32)
    next_border_points = cp.zeros(num_points, dtype=cp.int32)
    
    current_border_points[point_index] = 1
    
    while cp.any(current_border_points).get():

        expand_cluster_kernel_gpu(points, vector_degree, adjacent_indexes, adjacent_list, current_border_points, next_border_points, cluster_id, labels, min_pts)
        cp.cuda.Device().synchronize()
        
        current_border_points, next_border_points = next_border_points, current_border_points
        next_border_points.fill(0)

def expand_cluster_kernel_gpu(points, vector_degree, adjacent_indexes, adjacent_list, current_border_points, next_border_points, cluster_id, labels, min_pts):
    num_points = len(points) // 2

    threads_per_block = 256
    blocks_per_grid = (num_points + (threads_per_block - 1)) // threads_per_block
    kernel_code =  r'''
    extern "C" __global__
    void expand_cluster_kernel(const int *vector_degree, const int *adjacent_indexes, const int *adjacent_list, 
                                int *current_border_points, int *next_border_points, const int num_points, const int cluster_id, int *labels, const int min_pts) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_points) {
            if (current_border_points[idx] == 1) {
                int old_label = atomicCAS(&labels[idx], -1, cluster_id);
                if (old_label == -1) {
                    int start_idx = adjacent_indexes[idx];
                    int degree = vector_degree[idx];
                    if(degree + 1 >= min_pts) { // Only expand if core point
                        for (int j = start_idx; j < start_idx + degree; j++) {
                            int neighbor = adjacent_list[j];
                            atomicMax(&next_border_points[neighbor], 1);
                        }
                    }
                }
            }
        }
    }
    '''
    module = cp.RawModule(code=kernel_code)
    expand_cluster_kernel_func = module.get_function('expand_cluster_kernel')
    expand_cluster_kernel_func((blocks_per_grid,), (threads_per_block,), (vector_degree, adjacent_indexes, adjacent_list, current_border_points, next_border_points, num_points, cluster_id, labels,min_pts))
    

def build_graph(points, eps,min_pts):
    num_points = len(points) // 2
    
    vector_degree = cp.zeros(num_points, dtype=cp.int32) # Vertices degree
    vector_type = cp.zeros(num_points, dtype=cp.int32) # Vertex type: core or not
    vector_degree,vector_type = neigbours_count(points, eps, vector_degree, vector_type,min_pts)
    
    # Calcular adjacent_indexes usando prefix sum excluyente
    adjacent_indexes = cp.zeros(num_points, dtype=cp.int32)
    if num_points > 1:
        adjacent_indexes[1:] = cp.cumsum(vector_degree[:-1])
    
    total_neighbors = int(vector_degree.sum())
    
    adjacent_list = build_adjacency_list_from_indexes(points, eps, vector_degree, adjacent_indexes)
    
    return vector_degree, vector_type, adjacent_indexes, adjacent_list

def neigbours_count(points, eps, vector_degree, vector_type, min_pts):
    num_points = len(points) // 2
    
    threads_per_block = 256
    blocks_per_grid = (num_points + (threads_per_block - 1)) // threads_per_block
    
    eps2 = eps * eps
    
    kernel_code =  f'''
    
    extern "C" __global__
    void neigbours_count(const int *points, int *vector_degree, int *vector_type, const int num_points, const double eps2, const int min_pts) {{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_points) {{
            int x1 = points[idx * 2];
            int y1 = points[idx * 2 + 1];
            int count = 0;
            for (int j = 0; j < num_points; j++) {{
                if (j == idx) continue; // Skip self-loop
                int x2 = points[j * 2];
                int y2 = points[j * 2 + 1];
                double dx = (double) x2 - x1;
                double dy = (double) y2 - y1;
                double distance = (dx * dx + dy * dy);
                if (distance <= eps2) {{
                    count++;
                }}
            }}
            vector_degree[idx] = count ; // Store the neighbor count
            if (count + 1 >= min_pts) {{
                vector_type[idx] = 1; // Core point
            }} else {{
                vector_type[idx] = -1;
            }}
        }}
    }}
    '''
    module = cp.RawModule(code=kernel_code)
    neigbours_count_kernel = module.get_function('neigbours_count')
    neigbours_count_kernel((blocks_per_grid,), (threads_per_block,), (points, vector_degree, vector_type, cp.int32(num_points), cp.float64(eps2), cp.int32(min_pts)))
    return vector_degree, vector_type

def build_adjacency_list_from_indexes(points, eps, vector_degree, adjacent_indexes):
    num_points = len(points) // 2
    
    total_neighbors = int(vector_degree.sum())
    if total_neighbors == 0:
        return cp.zeros(0, dtype=cp.int32)
    
    adjacent_list = cp.zeros(total_neighbors, dtype=cp.int32)
    
    threads_per_block = 256
    blocks_per_grid = (num_points + (threads_per_block - 1)) // threads_per_block
    
    eps2 = eps * eps
    kernel_code = f'''
    extern "C" __global__
    void build_adjacency_list_from_indexes(const int *points, const int *vector_degree,const int *adjacent_indexes, int *adjacent_list, const int num_points, const double eps2) {{
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_points) {{
            int x1 = points[idx * 2];
            int y1 = points[idx * 2 + 1];
            
            int start_idx = adjacent_indexes[idx];
            int degree = vector_degree[idx];
            int count = 0;
            
            for (int j = 0; j < num_points; j++) {{
                if (j == idx) continue; // Skip self-loop
                int x2 = points[j * 2];
                int y2 = points[j * 2 + 1];
                double dx = (double) x2 - x1;
                double dy = (double) y2 - y1;
                double distance = (dx * dx + dy * dy);
                
                if (distance <= eps2) {{
                    adjacent_list[start_idx + count] = j;
                    count++;
                }}  
            }}
        }}
    }}
    '''
    module = cp.RawModule(code=kernel_code)
    build_adjacency_list_kernel = module.get_function('build_adjacency_list_from_indexes')
    build_adjacency_list_kernel((blocks_per_grid,), (threads_per_block,), (points, vector_degree, adjacent_indexes, adjacent_list, cp.int32(num_points), cp.float64(eps2)))
    return adjacent_list


def paint_clusters(image, points, labels, cluster_count, color_marker):
    
    num_points = len(labels)
    if len(points) != num_points * 2:
        raise ValueError(f"Length of points ({len(points)}) does not match 2 * length of labels ({2 * num_points})")
    
    points_2d = points.reshape(-1, 2) # Cambia la forma de points a (N, 2)

    # Use color_marker to determine background and foreground
    # image es un array de CuPy, convertir a numpy para matplotlib
    image_np = cp.asnumpy(image)
    if color_marker == 1:
        base = (1 - image_np) * 255 # white background
    else:
        base = image_np * 255 # black background

    # Create an RGB image from the base
    image_rgb = np.stack([base, base, base], axis=-1).astype(np.uint8)

    # If no clusters or no points, return the base image
    if cluster_count == 0 or num_points == 0:
        return image_rgb

    # Separate noise and clusters
    noise_mask = labels == -1 # Asumiendo -1 como noise
    cluster_mask = labels >= 0 # Asumiendo >= 0 como clusters

    # Paint noise in black
    if np.any(noise_mask):
        noise_pts = points_2d[noise_mask] # Usar points_2d
        if len(noise_pts) > 0:
            y_coords = np.clip(noise_pts[:, 1], 0, image_rgb.shape[0] - 1)
            x_coords = np.clip(noise_pts[:, 0], 0, image_rgb.shape[1] - 1)
            image_rgb[y_coords, x_coords] = [0, 0, 0]  # noise in black

    # Paint clusters with colors
    if np.any(cluster_mask):
        cluster_pts = points_2d[cluster_mask] # Usar points_2d
        cluster_labels = labels[cluster_mask]
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)

        # Generate distinct colors (excluding black and white)
        if n_clusters > 1:
            hues = np.linspace(0, 1, n_clusters) # Usar todos los colores disponibles
            colors = (plt.cm.hsv(hues)[:, :3] * 255).astype(np.uint8)
        else:
            colors = np.array([[255, 165, 0]], dtype=np.uint8) # Naranja si solo hay un cluster

        # Map labels to colors
        # Crear un array de colores para cada punto de cluster
        cluster_colors = np.zeros((len(cluster_labels), 3), dtype=np.uint8)
        for i, label in enumerate(unique_labels):
             mask_for_label = cluster_labels == label
             color_idx = np.where(unique_labels == label)[0][0]
             cluster_colors[mask_for_label] = colors[color_idx]

        # Assign colors to image
        if len(cluster_pts) > 0:
            y_coords = np.clip(cluster_pts[:, 1], 0, image_rgb.shape[0] - 1)
            x_coords = np.clip(cluster_pts[:, 0], 0, image_rgb.shape[1] - 1)
            image_rgb[y_coords, x_coords] = cluster_colors

    return image_rgb

if __name__ == "__main__":
    print("gpu_dbscan: Starting GPU DBSCAN clustering")

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
    points = points.reshape(-1, 2)
    points = points[cp.lexsort(cp.stack((points[:,1], points[:,0])))]
    points = points.ravel()

    points_cpu = cp.asnumpy(points)
    print(f"gpu_dbscan: Number of points to cluster: {len(points_cpu)//2}")
    
    timePoints = time.time() # Time after points extraction
    print("gpu_dbscan: TimePoints = ", timePoints - start)
    
    eps = get_epsilon(points, k=5,std_scale=std_scale)
    timeEpsilon = time.time() # Time after epsilon calculation
    print("gpu_dbscan: TimeEpsilon = ", timeEpsilon - timePoints)

    labels,cluster_count = dbscan(points, eps, min_pts=5)
    print(f"Number of clusters found: {cluster_count}")
    
    # End timing
    end = time.time()
    print("cpu_dbscan: TimeDBSCAN = ", end - timeEpsilon)
    print("cpu_dbscan: Final time = ", end - start)
    
    # Paint clusters on the image
    image_colored = paint_clusters(image_bw, points_cpu, labels, cluster_count, color_marker)
    save_image(image_colored) # Save the colored image

