#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

using namespace std;

struct Point3D {
  double x, y, z; // coordinates
  int cluster;    // no default cluster
  double minDist; // default infinite distance to the nearest cluster

  Point3D()
      : x(0.0), y(0.0), z(0.0), cluster(-1),
        minDist(numeric_limits<double>::max()) {}
  Point3D(double x, double y, double z)
      : x(x), y(y), z(z), cluster(-1), minDist(numeric_limits<double>::max()) {}

  /**
   * Computes the (square) Euclidean distance between this point and another
   */
  double distance(Point3D p) {
    return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y) +
           (p.z - z) * (p.z - z);
  }
};

/**
 * Reads in the data.csv file into a vector of points
 * @return vector of points
 *
 */
vector<Point3D> readcsv();
/**
 * Perform k-means clustering
 * @param points - pointer to vector of points
 * @param epochs - number of k means iterations
 * @param k - the number of initial centroids
 */
void kMeansClustering(vector<Point3D> *points, int epochs, int k);

void saveOutputs(vector<Point3D> *points, string filename);

void performSerial(int epoch, int clusterCount);