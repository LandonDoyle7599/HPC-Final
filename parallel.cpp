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
void kMeansClusteringParallelCPU(vector<Point3D> *points, int numEpochs, vector<Point3D> *centroids)
{
  int threads = omp_get_max_threads();
  // Repeat over epochs to converge the centroids

  for (int i = 0; i < numEpochs; ++i)
  {
    Point3D p;
    // For each centroid, compute distance from centroid to each point and update point's cluster if necessary
#pragma omp parallel for num_threads(threads) default(none) shared(points, centroids) private(p)
    for (int j = 0; j < centroids->size(); ++j)
    {
      // Check the distance from each point to the centroid and update each point if necessary
      for (int pointIndex = 0; pointIndex < points->size(); ++pointIndex)
      {
        Point3D p = points->at(pointIndex);
        double dist = centroids->at(j).distance(p);
        // we only want one thread updating the points at a time
        if (dist < p.minDist)
#pragma omp critical
        {
          p.minDist = dist;
          p.cluster = j;
          points->at(pointIndex) = p;
        }
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
