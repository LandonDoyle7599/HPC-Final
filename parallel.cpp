// #include "serial.hpp" // uncomment this to run individually
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>
#include <omp.h>
#include <vector>

using namespace std;

/**
 * Perform k-means clustering
 * @param points - pointer to vector of points
 * @param numEpochs - number of k means iterations
 * @param centroids - pointer to vector of centroids
 */
void kMeansClusteringParallelCPU(vector<Point3D> *points, int numEpochs, vector<Point3D> *centroids, int numThreads)
{
  int numPoints = points->size();
  int numCentroids = centroids->size();
  float minDistance;
  int clusterID;
  float distance;
  int epoch;
  int i;
  int j;

#pragma omp parallel num_threads(numThreads) default(none) shared(points, centroids, numEpochs, numPoints, numCentroids) private(epoch, i, j, minDistance, clusterID, distance)
// # pragma omp parallel num_threads(numThreads)
{
  // Repeat over epochs to converge the centroids
  for (epoch = 0; epoch < numEpochs; ++epoch)
  {
    #pragma omp for
    for (i = 0; i < numPoints; ++i)
    {
      minDistance = calculateDistanceSerial(points->at(i).x, points->at(i).y, points->at(i).z, centroids->at(0).x, centroids->at(0).y, centroids->at(0).z);
      clusterID = 0;
      for (j = 1; j < numCentroids; ++j)
      {
        distance = calculateDistanceSerial(points->at(i).x, points->at(i).y, points->at(i).z, centroids->at(j).x, centroids->at(j).y, centroids->at(j).z);
        if (distance < minDistance)
        {
          minDistance = distance;
          clusterID = j;
        }
      }
      // Update the cluster id and minimum distance.
      # pragma omp critical 
      {
        points->at(i).cluster = clusterID;
      }
    }

// Update the centroids
// We only want the root thread to update the centroids
# pragma omp barrier
#pragma omp master 
  {
    updateCentroidData(points, centroids, numCentroids);
  }
  # pragma omp barrier
}
}
}

void performParallel(int numEpochs, vector<Point3D> *centroids, vector<Point3D> *points, string filename, int numThreads)
{
  // Time code: https://stackoverflow.com/questions/21856025/getting-an-accurate-execution-time-in-c-micro-seconds
  // create centroids
  cout << "\tEntering the k means computation" << endl;
  auto start_time = std::chrono::high_resolution_clock::now();
  kMeansClusteringParallelCPU(points, numEpochs, centroids, numThreads); // K-means clustering on the points.
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  printStats(numEpochs, centroids->size(), points, duration.count());
  saveOutputs(points, filename);
}
