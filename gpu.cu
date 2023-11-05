#include "serial.cpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
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
__global__ void kMeansClusteringKernel(Point3D *points, Point3D *centroids, int nPoints, int k) {
    // Get thread (point) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= nPoints) {
        return;
    }

    // Find the closest centroid to this point
    float minDist = numeric_limits<float>::max();
    int clusterId = 0;
    for (int i = 0; i < k; ++i) {
        float dist = calculateDistance(points[tid], centroids[i]);
        if (dist < minDist) {
            minDist = dist;
            clusterId = i;
        }
    }

    // Update cluster id and minimum distance for this point
    points[tid].cluster = clusterId;
    points[tid].minDist = minDist;
}

/**
 * Perform k-means clustering with a GPU
 * @param points - pointer to vector of points
 * @param epochs - number of k means iterations
 * @param k - the number of initial centroids
 */
void kMeansClusteringGPU(vector<Point3D> *points, int epochs, int k)
{
    // Randomly initialize centroids
  vector<Point3D> centroids;
  srand(time(0));
  centroids.reserve(k);
  for (int i = 0; i < k; ++i)
  {
    centroids.push_back(points->at(rand() % points->size()));
  }

  for (int i = 0; i < epochs; ++i)
  {
    // For each centroid, compute distance from centroid to each point
    // and update point's cluster if necessary

    // Allocate memory on GPU
    Point3D *d_points;
    Point3D *d_centroids;
    cudaMalloc(&d_points, points->size() * sizeof(Point3D));
    cudaMalloc(&d_centroids, centroids.size() * sizeof(Point3D));

    // Copy data to GPU
    cudaMemcpy(d_points, points->data(), points->size() * sizeof(Point3D), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids.data(), centroids.size() * sizeof(Point3D), cudaMemcpyHostToDevice);

    // Run kernel
    int blockSize = 1024;
    int gridSize = (int)ceil((float)points->size() / blockSize);
    kMeansClusteringKernel<<<gridSize, blockSize>>>(d_points, d_centroids, points->size(), k);

    // Copy data back to CPU
    cudaMemcpy(points->data(), d_points, points->size() * sizeof(Point3D), cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids.data(), d_centroids, centroids.size() * sizeof(Point3D), cudaMemcpyDeviceToHost);

    // Free memory on GPU
    cudaFree(d_points);
    cudaFree(d_centroids);

    // Update centroids
    for (int i = 0; i < k; ++i)
    {
      int nPoints = 0;
      float sumX = 0;
      float sumY = 0;
      float sumZ = 0;
      for (int j = 0; j < points->size(); ++j)
      {
        if (points->at(j).cluster == i)
        {
          sumX += points->at(j).x;
          sumY += points->at(j).y;
          sumZ += points->at(j).z;
          nPoints++;
        }
      }
      centroids[i].x = sumX / nPoints;
      centroids[i].y = sumY / nPoints;
      centroids[i].z = sumZ / nPoints;
    }

    // Print progress
    cout << "Epoch " << i + 1 << " complete" << endl;
}
}

void performGPUKMeans(int epochs, int k)
{
    // First we use the same readcsv function as in serial.cpp. TODO: Use the parallel version of this to read in the values
    vector<Point3D> points = readcsv();
    kMeansClusteringGPU(&points, epochs, k);
    saveOutputs(points, "single-gpu.csv");
}

