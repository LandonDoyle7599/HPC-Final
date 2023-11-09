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
void kMeansClusteringParallel(vector<Point3D> *points, int numEpochs, vector<Point3D> *centroids)
{
  for (int epoch = 0; epoch < numEpochs; ++epoch)
  {
#pragma omp parallel for
    for (int i = 0; i < points->size(); ++i)
    {
      Point3D &p = (*points)[i];
      int clusterId = 0;
      double minDist = centroids->at(0).distance(p);

      for (int j = 1; j < centroids->size(); ++j)
      {
        double dist = centroids->at(j).distance(p);
        if (dist < minDist)
        {
          minDist = dist;
          clusterId = j;
        }
      }

#pragma omp critical
      {
        p.minDist = minDist;
        p.cluster = clusterId;
      }
    }

    // Update the centroids
    updateCentroidData(points, centroids, centroids->size());
  }
}

void performParallel(int numEpochs, vector<Point3D> *centroids, vector<Point3D> *points, string filename)
{
  // Time code: https://stackoverflow.com/questions/21856025/getting-an-accurate-execution-time-in-c-micro-seconds
  // create centroids
  cout << "Entering the k means computation" << endl;
  auto start_time = std::chrono::high_resolution_clock::now();
  kMeansClusteringParallelCPU(points, numEpochs, centroids); // K-means clustering on the points.
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
