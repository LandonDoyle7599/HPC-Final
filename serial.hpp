#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

using namespace std;

/**
 * Represents a 3D point and its corresponding closest cluster
 */
struct Point3D
{
  double x, y, z; // coordinates
  int cluster;    // no default cluster

  Point3D()
      : x(0.0), y(0.0), z(0.0), cluster(-1) {}
  Point3D(double x, double y, double z)
      : x(x), y(y), z(z), cluster(-1) {}
};

/**
 * Saves the points to a csv file
 * @param points - pointer to vector of points
 * @param filename - name of file to save to
 */
void saveOutputs(vector<Point3D> *points, string filename)
{
  cout << "\tSaving Output" << endl;
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
 * Initializes the centroids
 * @param numCentroids - the number of initial centroids
 * @param points - pointer to vector of points
 * @param random - decides whether to randomly initalize the centeroids or not
 * @return vector of centroids
 */
vector<Point3D> initializeCentroids(int numCentroids, vector<Point3D> *points)
{
  // Randomly initialize centroids
  // The index of the centroid within the centroids vector represents the cluster label.
  vector<Point3D> centroids;
  srand(10);                       // seed the random number generator
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
  vector<int> nPoints(numCentroids, 0);
  vector<double> sumX(numCentroids, 0.0);
  vector<double> sumY(numCentroids, 0.0);
  vector<double> sumZ(numCentroids, 0.0);

  // Iterate over points to append data to centroids
  for (vector<Point3D>::iterator it = points->begin(); it != points->end(); ++it)
  {
    int clusterId = it->cluster;
    nPoints[clusterId] += 1;
    sumX[clusterId] += it->x;
    sumY[clusterId] += it->y;
    sumZ[clusterId] += it->z;
  }
  // Compute the new centroids
  for (vector<Point3D>::iterator c = centroids->begin(); c != centroids->end(); ++c)
  {
    int clusterId = c - centroids->begin();
    c->x = sumX[clusterId] / nPoints[clusterId];
    c->y = sumY[clusterId] / nPoints[clusterId];
    c->z = sumZ[clusterId] / nPoints[clusterId];
  }
}

/**
 *  Compares two files to see if they are equal
 * @param filename1 - name of first file
 * @param filename2 - name of second file
 * @param showDiff - whether to show the differences or not
 */
bool areFilesEqual(string filename1, string filename2, bool showDiff = false)
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
      counter++;
      flag = false;
      // Exit out early if we don't want to see the debugging of 5 lines
      if (!showDiff)
      {
        file1.close();
        file2.close();
        return false;
      }
      else if (counter < 5)
      {
        std::cout << "Difference in line " << lineNum << ":\n";
        std::cout << "File 1: " << line1 << "\n";
        std::cout << "File 2: " << line2 << "\n\n";
      }
    }
    lineNum++;
  }

  std::cout << "Total differences: " << counter << "\n";

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

/**
 * Reads in the data.csv file into a vector of points
 * @param filename - the name of the file to read
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
 * Prints the stats of the program
 * @param numEpochs - number of k means iterations
 * @param numCentroids - the number of initial centroids
 * @param points - pointer to vector of points
 * @param duration - the time it took to run the program
 */
void printStats(int numEpochs, int numCentroids, vector<Point3D> *points, long duration)
{
  cout << "\n---- STATS ----" << endl;
  cout << "Points: " << points->size() << endl;
  cout << "Epochs " << numEpochs << endl;
  cout << "Clusters: " << numCentroids << endl;
  cout << "Time: " << duration << endl;
  cout << "--------------\n"
       << endl;
}

/**
 * Gets the epochs and centroids from the command line
 * @param argc - number of command line arguments
 * @param argv - array of command line arguments
 * @param numEpochs - number of k means iterations
 * @param numCentroids - the number of initial centroids
 */
void getEpochsCentroids(int argc, char *argv[], int &numEpochs, int &numCentroids)
{
  if (argc < 3)
  {
    cout << "Usage: " << argv[0] << " <numEpochs> <numCentroids> " << endl;
    exit(0);
  }
  numEpochs = atoi(argv[1]);
  numCentroids = atoi(argv[2]);
}