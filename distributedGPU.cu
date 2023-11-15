#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cmath>
#include <iostream>
using namespace std;


__device__ float calculateDistance(float x1, float y1, float z1, float x2, float y2, float z2){
    float x = abs(x1 - x2);
    float y = abs(y1 - y2);
    float z = abs(z1 - z2);
    return (x * x) + (y * y) + (z * z);
} 

__global__ void calculateKMean(double k_x[], double k_y[], double k_z[], double recv_x[], double recv_y[], double recv_z[], int assign[], int numLocalDataPoints, int numCentroids){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if we are out of bounds
    if (i > numLocalDataPoints){
        return;
    }
    double min_dist = calculateDistance(k_x[0], k_y[0], k_z[0], recv_x[i], recv_y[i], recv_z[i]);
    int clusterID = 0;
    for (int j = 1; j < numCentroids; ++j)
    // Find the closest centroid
    {
        double temp_dist = calculateDistance(k_x[j], k_y[j], k_z[j], recv_x[i], recv_y[i], recv_z[i]);

        if (temp_dist < min_dist)
        {
            min_dist = temp_dist;
            clusterID = j;
        }
    }
    // Update the assignment
    assign[i] = clusterID;
}

extern "C" {
    void launchCalculateKMean(double k_x[], double k_y[], double k_z[], double recv_x[], double recv_y[], double recv_z[], int assign[], int numLocalDataPoints, int numCentroids){
        int threadsPerBlock = 256;
        int blocksPerGrid = (numLocalDataPoints / threadsPerBlock) + 1;
        // Allocate memory on the GPU
        double *d_k_x, *d_k_y, *d_k_z, *d_recv_x, *d_recv_y, *d_recv_z;
        int *d_assign;
        cudaMalloc((void **)&d_k_x, numCentroids * sizeof(double));
        cudaMalloc((void **)&d_k_y, numCentroids * sizeof(double));
        cudaMalloc((void **)&d_k_z, numCentroids * sizeof(double));
        cudaMalloc((void **)&d_recv_x, numLocalDataPoints * sizeof(double));
        cudaMalloc((void **)&d_recv_y, numLocalDataPoints * sizeof(double));
        cudaMalloc((void **)&d_recv_z, numLocalDataPoints * sizeof(double));
        cudaMalloc((void **)&d_assign, numLocalDataPoints * sizeof(int));
        // Copy data to the GPU
        cudaMemcpy(d_k_x, k_x, numCentroids * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_k_y, k_y, numCentroids * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_k_z, k_z, numCentroids * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_recv_x, recv_x, numLocalDataPoints * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_recv_y, recv_y, numLocalDataPoints * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_recv_z, recv_z, numLocalDataPoints * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_assign, assign, numLocalDataPoints * sizeof(int), cudaMemcpyHostToDevice);
        
        // Launch the kernel
        calculateKMean<<<blocksPerGrid, threadsPerBlock>>>(k_x, k_y, k_z, recv_x, recv_y, recv_z, assign, numLocalDataPoints, numCentroids);
        cudaDeviceSynchronize();
        // Copy the result back
        cudaMemcpy(assign, d_assign, numLocalDataPoints * sizeof(int), cudaMemcpyDeviceToHost);
        // Free the memory
        cudaFree(d_k_x);
        cudaFree(d_k_y);
        cudaFree(d_k_z);
        cudaFree(d_recv_x);
        cudaFree(d_recv_y);
        cudaFree(d_recv_z);
        cudaFree(d_assign);
    }
}
