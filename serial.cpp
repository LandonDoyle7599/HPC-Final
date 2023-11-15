#include "serial.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>

using namespace std;

/**
 * Calculates the distance between two points
 * @param p_x - x coordinate of point
 * @param p_y - y coordinate of point
 * @param p_z - z coordinate of point
 * @param k_x - x coordinate of centroid
 * @param k_y - y coordinate of centroid
 * @param k_z - z coordinate of centroid
 * @return the distance between the two points
 */
float calculateDistanceSerial(float p_x, float p_y, float p_z, float k_x, float k_y, float k_z)
{
  float dx = p_x - k_x;
  float dy = p_y - k_y;
  float dz = p_z - k_z;
  return (dx * dx) + (dy * dy) + (dz * dz);
}

/**
 * Perform k-means clustering
 * @param points - pointer to vector of points
 * @param numEpochs - number of k means iterations
 * @param centroids - pointer to vector of centroids
 */
void kMeansClusteringSerial(vector<Point3D> *points, int numEpochs, vector<Point3D> *centroids)
{
  // Repeat over epochs to converge the centroids
  for (int i = 0; i < numEpochs; ++i)
  {
    // For each centroid, compute distance from centroid to each point
    // and update point's cluster if necessary
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
      points->at(j).cluster = clusterID;
    }
    // Update the centroids
    updateCentroidData(points, centroids, centroids->size());
  }
}

void performSerial(int numEpochs, vector<Point3D> *centroids, vector<Point3D> *points, string filename)
{
  // Time code: https://stackoverflow.com/questions/21856025/getting-an-accurate-execution-time-in-c-micro-seconds
  // create centroids
  cout << "Performing Serial CPU" << endl;
  auto start_time = std::chrono::high_resolution_clock::now();
  kMeansClusteringSerial(points, numEpochs, centroids); // K-means clustering on the points.
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  printStats(numEpochs, centroids->size(), points, duration.count());
  saveOutputs(points, filename);
}
