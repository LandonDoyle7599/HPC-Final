#include "serial.hpp"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

/**
 * Reads in the data.csv file into a vector of points
 * @return vector of points
 *
 */
vector<Point3D> readcsv(string filename)
{
  vector<Point3D> points;
  string line;
  ifstream file(filename);
  if (!file.is_open())
    cout << "Failed to open file\n";
  while (getline(file, line))
  {
    stringstream lineStream(line);
    string bit;
    double x, y, z;
    getline(lineStream, bit, ',');
    x = stof(bit);
    getline(lineStream, bit, ',');
    y = stof(bit);
    getline(lineStream, bit, '\n');
    z = stof(bit);
    points.push_back(Point3D(x, y, z));
  }
  return points;
}

/**
 * Initializes the centroids
 * @param numCentroids - the number of initial centroids
 * @param points - pointer to vector of points
 * @return vector of centroids
 */
vector<Point3D> initializeCentroids(int numCentroids, vector<Point3D> *points)
{
  // Randomly initialize centroids
  // The index of the centroid within the centroids vector represents the cluster label.
  vector<Point3D> centroids;
  srand(time(0));
  centroids.reserve(numCentroids); // create space in memory for specified number of centroids
  for (int i = 0; i < numCentroids; ++i)
  {
    centroids.push_back(points->at(rand() % points->size()));
  }
  return centroids;
}

/**
 * Updates the centroid data based on the points
 * @param points - pointer to vector of points
 * @param centroids - pointer to vector of centroids
 * @param numCentroids - the number of initial centroids
 */

void updateCentroidData(vector<Point3D> *points, vector<Point3D> *centroids, int numCentroids)
{
  // Create vectors to keep track of data needed to compute means
  vector<int> nPoints;
  vector<double> sumX, sumY;
  for (int j = 0; j < numCentroids; ++j)
  {
    nPoints.push_back(0);
    sumX.push_back(0.0);
    sumY.push_back(0.0);
  }
  // Iterate over points to append data to centroids
  for (vector<Point3D>::iterator it = points->begin(); it != points->end(); ++it)
  {
    int clusterId = it->cluster;
    nPoints[clusterId] += 1;
    sumX[clusterId] += it->x;
    sumY[clusterId] += it->y;

    it->minDist = numeric_limits<float>::max(); // reset distance
  }
  // Compute the new centroids
  for (vector<Point3D>::iterator c = begin(centroids); c != end(centroids); ++c)
  {
    int clusterId = c - begin(centroids);
    c->x = sumX[clusterId] / nPoints[clusterId];
    c->y = sumY[clusterId] / nPoints[clusterId];
  }
}

/**
 * Saves the points to a csv file
 * @param points - pointer to vector of points
 * @param filename - name of file to save to
 */
void saveOutputs(vector<Point3D> *points, string filename)
{
  ofstream myfile;
  myfile.open(filename);
  myfile << "x,y,z,c" << endl;
  for (vector<Point3D>::iterator it = points->begin(); it != points->end();
       ++it)
  {
    myfile << it->x << "," << it->y << "," << it->z << "," << it->cluster
           << endl;
  }
  myfile.close();
}

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

    updateCentroidData(points, &centroids, numCentroids);
  }
}

void performSerial(int numEpochs, int clusterCount)
{
  vector<Point3D> points = readcsv("song_data.csv");
  kMeansClustering(&points, numEpochs, clusterCount); // K-means clustering on the points.
  saveOutputs(&points, "serialOutput.csv");
}
