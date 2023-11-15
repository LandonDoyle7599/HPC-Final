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
  // Repeat over epochs to converge the centroids
  for (int i = 0; i < numEpochs; ++i)
  {
#pragma omp parallel for num_threads(numThreads)
    for (int j = 0; j < points->size(); ++j)
    {
      float minDistance = calculateDistanceSerial(points->at(j).x, points->at(j).y, points->at(j).z, centroids->at(0).x, centroids->at(0).y, centroids->at(0).z);
      int clusterID = 0;
      for (int k = 1; k < centroids->size(); ++k)
      {
        float distance = calculateDistanceSerial(points->at(j).x, points->at(j).y, points->at(j).z, centroids->at(k).x, centroids->at(k).y, centroids->at(k).z);
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
    updateCentroidData(points, centroids, centroids->size());
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

// Uncomment this to run the serial code standalone
// int main()
// {
//   // Read in the data
//   cout << "Reading the csv" << endl;
//   vector<Point3D> points = readcsv("song_data.csv");
//   int numEpochs = 100;
//   int numCentroids = 6;
//   // Initialize the centroids
//   vector<Point3D> centroids = initializeCentroids(numCentroids, &points, true);
//   // Perform it
//   performParallel(numEpochs, &centroids, &points, "parallel-cpu.csv");
// }
