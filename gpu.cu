#include "serial.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <cmath>

using namespace std;

 // Define a GPU device function to calculateDistance
 __device__ float calculateDistance(Point3D point, Point3D centroid) {
    float dx = point.x - centroid.x;
    float dy = point.y - centroid.y;
    float dz = point.z - centroid.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

// Define a GPU kernel to perform k-means clustering
__global__ void kMeansClusteringKernel(Point3D *points, Point3D *centroids,int* clusterAssignments, int nPoints, int numCentroids) {
    // Get thread ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Exit if we are out of bounds
    if (tid >= nPoints) {
        return;
    }
    // Find the closest centroid to this point
    float minDist = calculateDistance(points[tid], centroids[0]); // setup first point
    int clusterId = 0; // setup first cluster id
    for (int i = 1; i < numCentroids; ++i) {
        // Point3D p = points[i];
        float dist = calculateDistance(points[tid], centroids[i]); // calculate distance between point and centroid with GPU function
        if (dist < minDist) {
            minDist = dist;
            clusterId = i;
        }
    }
    // Update cluster id and minimum distance for this point
    clusterAssignments[tid] = clusterId; // clusterAssignments is a global array of ints [nPoints]
    points[tid].cluster = clusterId;
    points[tid].minDist = minDist;
}

/**
 * Perform k-means clustering with a GPU
 * @param points - pointer to vector of points
 * @param numEpochs - number of k means iterations
 * @param k - the number of initial centroids
 */
void kMeansClusteringGPU(vector<Point3D> *points, int numEpochs, int numCentroids)
{
  // Initialize centroids
  vector<Point3D> centroids = initializeCentroids(numCentroids, points);

  // Run k-means clustering over number of numEpochs to converge the centroids
  for (int i = 0; i < numEpochs; ++i)
  {
    // Allocate memory on GPU
    Point3D *d_points;
    Point3D *d_centroids;
    int* d_clusterAssignments;

    cudaMalloc(&d_points, points->size() * sizeof(Point3D));
    cudaMalloc(&d_centroids, centroids.size() * sizeof(Point3D));
    cudaMalloc(&d_clusterAssignments, points->size() * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_points, points->data(), points->size() * sizeof(Point3D), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids.data(), centroids.size() * sizeof(Point3D), cudaMemcpyHostToDevice);

    // Run kernel to compute distance from centroid to each point
    int threadsPerBlock = 256;
    int blocksPerGrid = (int)ceil((float)points->size() / threadsPerBlock);
    kMeansClusteringKernel<<<blocksPerGrid, threadsPerBlock>>>(d_points, d_centroids, d_clusterAssignments, points->size(), numCentroids);

    // Copy data back to CPU
    cudaMemcpy(points->data(), d_points, points->size() * sizeof(Point3D), cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids.data(), d_centroids, centroids.size() * sizeof(Point3D), cudaMemcpyDeviceToHost);

    // Free memory on GPU
    cudaFree(d_points);
    cudaFree(d_centroids);

    // Update centroids
    updateCentroidData(points, &centroids, numCentroids);
  }
}

void performGPUKMeans(int numEpochs, int numCentroids)
{
    // First we use the same readcsv function as in serial.cpp. TODO: Use the parallel version of this to read in the values
    cout << "Reading the csv" << endl;
    vector<Point3D> points = readcsv("song_data.csv");
    cout << "Entering the k means computation" << endl;
    kMeansClusteringGPU(&points, numEpochs, numCentroids);
    cout << "Saving the output" << endl;
    saveOutputs(&points, "single-gpu.csv");
}

// Use this to run the program and compare outputs
int main() {
  // performGPUKMeans(100, 6);
  bool res = areFilesEqual("single-gpu.csv", "./persistedData/serialOutput.csv", true);
  std::cout << "Testing: " <<  res << std::endl;
}


