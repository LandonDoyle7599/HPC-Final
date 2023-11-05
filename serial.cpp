#include "serial.hpp"

using namespace std;

/**
 * Reads in the data.csv file into a vector of points
 * @return vector of points
 *
 */
vector<Point3D> readcsv()
{
  vector<Point3D> points;
  string line;
  // TODO: Make this a relative path
  ifstream file("C:\\Users\\lando\\CLionProjects\\HPCFinal\\song_data.csv");
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
 * Perform k-means clustering
 * @param points - pointer to vector of points
 * @param epochs - number of k means iterations
 * @param k - the number of initial centroids
 */
void kMeansClustering(vector<Point3D> *points, int epochs, int k)
{
  // Randomly initialize centroids
  // The index of the centroid within the centroids vector
  // represents the cluster label.
  vector<Point3D> centroids;
  srand(time(0));
  centroids.reserve(k);
  for (int i = 0; i < k; ++i)
  {
    centroids.push_back(points->at(rand() % points->size()));
  }

  for (int i = 0; i < epochs; ++i)
  {
    // For each centroid, compute distance from centroid to each point
    // and update point's cluster if necessary
    for (vector<Point3D>::iterator c = begin(centroids); c != end(centroids);
         ++c)
    {
      int clusterId = c - begin(centroids);

      for (vector<Point3D>::iterator it = points->begin(); it != points->end();
           ++it)
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

    // Create vectors to keep track of data needed to compute means
    vector<int> nPoints;
    vector<double> sumX, sumY;
    for (int j = 0; j < k; ++j)
    {
      nPoints.push_back(0);
      sumX.push_back(0.0);
      sumY.push_back(0.0);
    }

    // Iterate over points to append data to centroids
    for (vector<Point3D>::iterator it = points->begin(); it != points->end();
         ++it)
    {
      int clusterId = it->cluster;
      nPoints[clusterId] += 1;
      sumX[clusterId] += it->x;
      sumY[clusterId] += it->y;

      it->minDist = numeric_limits<float>::max(); // reset distance
    }
    // Compute the new centroids
    for (vector<Point3D>::iterator c = begin(centroids); c != end(centroids);
         ++c)
    {
      int clusterId = c - begin(centroids);
      c->x = sumX[clusterId] / nPoints[clusterId];
      c->y = sumY[clusterId] / nPoints[clusterId];
    }
  }

  // TODO: Make this a relative path
  saveOutputs(points, "C:\\Users\\lando\\CLionProjects\\HPCFinal\\serialOutput.csv");
}

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

void performSerial(int epoch, int clusterCount)
{
  vector<Point3D> points = readcsv();
  kMeansClustering(&points, epoch, clusterCount);
}