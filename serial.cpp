#include "serial.hpp"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;








/**
 * Perform k-means clustering
 * @param points - pointer to vector of points
 * @param numEpochs - number of k means iterations
 * @param numCentroids - the number of initial centroids
 */
void kMeansClustering(vector<Point3D> *points, int numEpochs, int numCentroids)
{

  vector<Point3D> centroids = initializeCentroids(numCentroids, points);
  // Repeat over epochs to converge the centroids
  
  for (int i = 0; i < numEpochs; ++i)
  {
    // For each centroid, compute distance from centroid to each point
    // and update point's cluster if necessary

    //TODO - parallelize this loop with openMP and distributed on MPI
    for (vector<Point3D>::iterator c = begin(centroids); c != end(centroids); ++c)
    {
      int clusterId = c - begin(centroids);

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
    updateCentroidData(points, &centroids, numCentroids);
  }
}

void performSerial(int numEpochs, int clusterCount)
{
  cout << "Reading the csv" << endl;
  vector<Point3D> points = readcsv("song_data.csv");
  cout << "Entering the k means computation" << endl;
  kMeansClustering(&points, numEpochs, clusterCount); // K-means clustering on the points.
  cout << "Saving the output" << endl;
  saveOutputs(&points, "serialOutput.csv");
}
