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
 * Updates the centroid data based on the points
 * @param points - pointer to vector of points
 * @param centroids - pointer to vector of centroids
 * @param numCentroids - the number of initial centroids
 */
void parallelUpdateCentroidData(vector<Point3D> *points, vector<Point3D> *centroids, int numCentroids)
{
  // Create vectors to keep track of data needed to compute means
  vector<int> nPoints(numCentroids, 0);
  vector<double> sumX(numCentroids, 0.0);
  vector<double> sumY(numCentroids, 0.0);

  // Parallelize the loop to accumulate data for centroid updates
#pragma omp parallel for
  for (int i = 0; i < points->size(); ++i)
  {
    int clusterId = (*points)[i].cluster;
#pragma omp atomic
    nPoints[clusterId] += 1;
#pragma omp atomic
    sumX[clusterId] += (*points)[i].x;
#pragma omp atomic
    sumY[clusterId] += (*points)[i].y;
    (*points)[i].minDist = numeric_limits<float>::max(); // reset distance
  }

  // Compute the new centroids
  for (int clusterId = 0; clusterId < numCentroids; ++clusterId)
  {
    centroids->at(clusterId).x = sumX[clusterId] / nPoints[clusterId];
    centroids->at(clusterId).y = sumY[clusterId] / nPoints[clusterId];
  }
}

/**
 * Perform k-means clustering
 * @param points - pointer to vector of points
 * @param numEpochs - number of k means iterations
 * @param centroids - pointer to vector of centroids
 */
void kMeansClusteringParallelCPU(vector<Point3D> *points, int numEpochs, vector<Point3D> *centroids, int numThreads)
{
  // Repeat over epochs to converge the centroids
  for (int epoch = 0; epoch < numEpochs; ++epoch)
  {
#pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < points->size(); ++i)
    {
      Point3D &p = (*points)[i]; // Get the point
      int clusterId = 0;
      double minDist = centroids->at(0).distance(p); // Get the distance to the first centroid

      for (int j = 1; j < centroids->size(); ++j) // Iterate over the rest of the centroids to see if it is closer to any others
      {
        double dist = centroids->at(j).distance(p);
        if (dist < minDist)
        {
          minDist = dist;
          clusterId = j;
        }
      }
// Update the cluster id and minimum distance. This is critical because we don't want the threads to overlap as they are writing to the same memory location
#pragma omp critical
      {
        p.minDist = minDist;
        p.cluster = clusterId;
      }
    }

    // Update the centroids
    parallelUpdateCentroidData(points, centroids, centroids->size());
  }
}

void performParallel(int numEpochs, vector<Point3D> *centroids, vector<Point3D> *points, string filename, int numThreads)
{
  // Time code: https://stackoverflow.com/questions/21856025/getting-an-accurate-execution-time-in-c-micro-seconds
  // create centroids
  cout << "Entering the k means computation" << endl;
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
