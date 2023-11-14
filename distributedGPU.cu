#include <cuda.h>
#include <cuda_runtime.h>

__global__ calculateKMean(double k_x[], double k_y[], double k_z[], double recv_x[], double recv_y[], double recv_z[], int assign[], int numLocalDataPoints, int numCentroids){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if we are out of bounds
    if (i > numLocalDataPoints){
        return;
    }
    double min_dist = numeric_limits<double>::max();
    int clusterID = 0;
    for (int j = 0; j < numCentroids; ++j)
    // Find the closest centroid
    {
        double x = abs(recv_x[i] - k_x[j]);
        double y = abs(recv_y[i] - k_y[j]);
        double z = abs(recv_z[i] - k_z[j]);
        double temp_dist = (x * x) + (y * y) + (z * z);

        if (temp_dist < min_dist)
        {
            min_dist = temp_dist;
            clusterID = j;
        }
    }
    // Update the assignment
    assign[i] = clusterID;
}