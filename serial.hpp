#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

using namespace std;

struct Point3D
{
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
  double distance(Point3D p)
  {
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

void performSerial(int numEpochs, int clusterCount);

vector<Point3D> initializeCentroids(int numCentroids, vector<Point3D> *points);

void updateCentroidData(vector<Point3D> *points, vector<Point3D> *centroids, int numCentroids);

bool areFilesEqual(string filename1, string filename2, bool showDiff)
{
  // Open the first CSV file
  std::ifstream file1(filename1);
  if (!file1.is_open())
  {
    std::cerr << "Error opening " << filename1 << std::endl;
    return false;
  }

  // Open the second CSV file
  std::ifstream file2(filename2);
  if (!file2.is_open())
  {
    std::cerr << "Error opening " << filename2 << std::endl;
    return false;
  }

  std::string line1, line2;
  int lineNum = 1;  // Line number for tracking differences
  bool flag = true; // let the parser compare the differences
  int counter = 0;

  while (getline(file1, line1) && getline(file2, line2))
  {
    if (line1 != line2)
    {
      flag = false;
      // Exit out early if we don't want to see the debugging of 5 lines
      if (!showDiff || counter > 5)
      {
        file1.close();
        file2.close();
        return false;
      }
      std::cout << "Difference in line " << lineNum << ":\n";
      std::cout << "File 1: " << line1 << "\n";
      std::cout << "File 2: " << line2 << "\n\n";
      counter++;
    }
    lineNum++;
  }

  // Check if one file has extra lines
  if (getline(file1, line1))
  {
    flag = false;
    while (getline(file1, line1))
    {
      std::cout << "Extra line in file 1 (line " << lineNum << "):\n";
      std::cout << "File 1: " << line1 << "\n\n";
      lineNum++;
    }
  }
  else if (getline(file2, line2))
  {
    while (getline(file2, line2))
    {
      flag = false;
      std::cout << "Extra line in file 2 (line " << lineNum << "):\n";
      std::cout << "File 2: " << line2 << "\n\n";
      lineNum++;
    }
  }
  file1.close();
  file2.close();
  return flag;
}