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
  float minDistance;
  int clusterID;
  float distance;
  int i;
  int j;
  int k;
  // Create a thread region
  omp_set_num_threads(numThreads);
#pragma omp parallel for default(none) shared(points, centroids) private(i, j, k, minDistance, clusterID, distance)
  // Repeat over epochs to converge the centroids
  for (i = 0; i < numEpochs; ++i)
  {
#pragma omp for
    for (j = 0; j < points->size(); ++j)
    {
      minDistance = calculateDistanceSerial(points->at(j).x, points->at(j).y, points->at(j).z, centroids->at(0).x, centroids->at(0).y, centroids->at(0).z);
      clusterID = 0;
      for (k = 1; k < centroids->size(); ++k)
      {
        distance = calculateDistanceSerial(points->at(j).x, points->at(j).y, points->at(j).z, centroids->at(k).x, centroids->at(k).y, centroids->at(k).z);
        if (distance < minDistance)
        {
          minDistance = distance;
          clusterID = k;
        }
      }
      // Update the cluster id and minimum distance.
      points->at(j).cluster = clusterID;
    }

// Update the centroids
// We only want the root thread to update the centroids
#pragma omp master {
    updateCentroidData(points, centroids, centroids->size());
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
