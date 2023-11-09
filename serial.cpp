#include "serial.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>

using namespace std;

/**
 * Perform k-means clustering
 * @param points - pointer to vector of points
 * @param numEpochs - number of k means iterations
 * @param numCentroids - the number of initial centroids
 */
void kMeansClustering(vector<Point3D> *points, int numEpochs, int numCentroids, vector<Point3D> *centroids)
{
  // Repeat over epochs to converge the centroids
  for (int i = 0; i < numEpochs; ++i)
  {
    // For each centroid, compute distance from centroid to each point
    // and update point's cluster if necessary

    // TODO - parallelize this loop with openMP and distributed on MPI
    for (vector<Point3D>::iterator c = begin(*centroids); c != end(*centroids); ++c)
    {
      int clusterId = c - begin(*centroids);

      for (vector<Point3D>::iterator it = points->begin(); it != points->end(); ++it)
      {
        Point3D p = *it;
        double dist = c->distance(p);
        if (dist < p.minDist)
        {
          p.minDist = dist;
          p.cluster = clusterId;
        }
        *it = p;
      }
    }
    // Update the centroids
    updateCentroidData(points, centroids, numCentroids);
  }
}

void performSerial(int numEpochs, int numCentroids)
{
  cout << "Reading the csv" << endl;
  vector<Point3D> points = readcsv("song_data.csv");
  // Time code: https://stackoverflow.com/questions/21856025/getting-an-accurate-execution-time-in-c-micro-seconds
  auto start_time = std::chrono::high_resolution_clock::now();
  // create centroids
  vector<Point3D> centroids = initializeCentroids(numCentroids, &points, true);
  cout << "Entering the k means computation" << endl;
  kMeansClustering(&points, numEpochs, numCentroids, &centroids); // K-means clustering on the points.

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  printStats(numEpochs, numCentroids, &points, duration);
  saveOutputs(&points, "serial-cpu.csv");
}

int main()
{
  int numEpochs = 100;
  int numCentroids = 6;
  performSerial(numEpochs, numCentroids);
}
