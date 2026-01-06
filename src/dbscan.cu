#include <cuda_runtime.h>
#include <stdio.h>

extern "C"{
    /**
     * @brief CUDA kernel for expanding clusters in the DBSCAN algorithm
     * 
     * This kernel processes border points and expands clusters by marking
     * neighbors of core points. It executes in parallel on the GPU.
     * 
     * @param vector_degree Degree of each point (number of neighbors)
     * @param adjacent_indexes Starting indexes in the adjacency list
     * @param adjacent_list Flat list of neighbors for all points
     * @param border_points Active points at the cluster frontier (1 = active, 0 = inactive)
     * @param num_points Total number of points
     * @param cluster_id ID of the current cluster being expanded
     * @param labels Cluster labels for each point (-1 = unvisited)
     * @param min_pts Minimum number of points to be considered a core point
     * @param active_flag Flag indicating if expansion occurred in this iteration
     */
    __global__ void expand_cluster_kernel(const int *vector_degree, const int *adjacent_indexes, const int *adjacent_list, 
                                int *border_points, const int num_points, const int cluster_id, int *labels, const int min_pts, int *active_flag) {
        // Calculate global thread index
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Check if index is within points range
        if (idx < num_points) {
            // If point is at border and hasn't been visited
            if (border_points[idx] != 0 && labels[idx] == -1) {
                labels[idx] = cluster_id;

                // Get neighbor information for current point
                int start_idx = adjacent_indexes[idx];
                int degree = vector_degree[idx];


                if(degree + 1 >= min_pts) { // Only expand if core point
                    *active_flag = 1;

                    // Expand to all neighbors
                    for (int j = start_idx; j < start_idx + degree; j++) {
                        int neighbor = adjacent_list[j];
                        border_points[neighbor] = 1;
                    }
                }
            }
        }
    }
    /**
     * @brief Main DBSCAN function that coordinates cluster expansion on GPU
     * 
     * This function manages GPU memory, launches kernels, and controls
     * iterative cluster expansion starting from unvisited core points.
     * 
     * @param points Point coordinates (not used in this function but kept for compatibility)
     * @param labels Output label array (-1 = noise, >=0 = cluster ID)
     * @param vector_degree Degree of each point
     * @param vector_type Type of each point (1 = core, -1 = non-core)
     * @param adjacent_indexes Starting indexes in adjacency list
     * @param adjacent_list Flat neighbor list
     * @param min_pts Minimum number of points to be core point
     * @param num_points Total number of points
     * @param adjacent_list_size Total size of adjacency list
     * @return int Total number of clusters found
     */
    int dbscan_core_cuda(const float *points, int *labels, const int *vector_degree, const int *vector_type, const int *adjacent_indexes,
        const int *adjacent_list, const int min_pts, const int num_points, const int adjacent_list_size) {

        int cluster_id = 0;
        
        // Declare pointers for GPU memory
        int *d_vector_degree, *d_adjacent_indexes, *d_adjacent_list;
        int *d_vector_type, *d_border_points, *d_labels, *d_active_flag;
        
        // 1. ALLOCATE GPU MEMORY
        cudaMalloc(&d_vector_degree, num_points * sizeof(int));
        cudaMalloc(&d_vector_type, num_points * sizeof(int));
        cudaMalloc(&d_adjacent_indexes, num_points * sizeof(int));
        cudaMalloc(&d_adjacent_list, adjacent_list_size * sizeof(int));
        cudaMalloc(&d_border_points, num_points * sizeof(int));
        cudaMalloc(&d_labels, num_points * sizeof(int));
        cudaMalloc(&d_active_flag, sizeof(int));
        
        // 2. COPY DATA FROM CPU TO GPU
        cudaMemcpy(d_vector_degree, vector_degree, num_points * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vector_type, vector_type, num_points * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_adjacent_indexes, adjacent_indexes, num_points * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_adjacent_list, adjacent_list, adjacent_list_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_labels, labels, num_points * sizeof(int), cudaMemcpyHostToDevice);

        // 3. KERNEL CONFIGURATION
        int threads_per_block = 256;
        int blocks_per_grid = (num_points + threads_per_block - 1) / threads_per_block;

        // 4. MAIN DBSCAN ALGORITHM
        for (int idx = 0; idx < num_points; idx++) {
            if (labels[idx] == -1 && vector_type[idx] == 1) { // Not visited and is core
                
                // 4.1 INITIALIZE CLUSTER FRONTIER
                // Clear all border points on GPU
                cudaMemset(d_border_points, 0, num_points * sizeof(int));
                // Activate initial point in border
                int initial_border = 1;
                cudaMemcpy(&d_border_points[idx], &initial_border, sizeof(int), cudaMemcpyHostToDevice);

                // 4.2 ITERATIVE CLUSTER EXPANSION
                int active = 1;
                while (active) {
                    // Reset activity flag for this iteration
                    int host_active_flag = 0;
                    cudaMemcpy(d_active_flag, &host_active_flag, sizeof(int), cudaMemcpyHostToDevice);

                    // 4.3 LAUNCH EXPANSION KERNEL
                    expand_cluster_kernel<<<blocks_per_grid, threads_per_block>>>(d_vector_degree, d_adjacent_indexes, d_adjacent_list,
                                                                                    d_border_points, num_points, cluster_id, d_labels, min_pts, d_active_flag);
                    // Wait for kernel to complete
                    cudaDeviceSynchronize();
                    
                    // 4.4 CHECK FOR EXPANSION
                    // Copy activity flag from GPU
                    cudaMemcpy(&host_active_flag, d_active_flag, sizeof(int), cudaMemcpyDeviceToHost);
                    active = (host_active_flag != 0); // Continue if there was activity
                }
                // 4.5 FINALIZE CURRENT CLUSTER
                cluster_id++;

                // 4.6 UPDATE LABELS FROM GPU
                // Copy updated labels to CPU for next iteration
                cudaMemcpy(labels, d_labels, num_points * sizeof(int), cudaMemcpyDeviceToHost);
            }
        }
        // 5. FREE GPU MEMORY
        cudaFree(d_vector_degree);
        cudaFree(d_vector_type);
        cudaFree(d_adjacent_indexes);
        cudaFree(d_adjacent_list);
        cudaFree(d_border_points);
        cudaFree(d_labels);
        cudaFree(d_active_flag);

        return cluster_id; // Return total number of clusters found
    }
} // End of extern "C" block

