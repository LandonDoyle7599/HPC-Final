#include "serial.cpp"
#include "parallel.cpp"
#include <iostream>
using namespace std;

void run(int numEpochs, int numCentroids, vector<Point3D> *points, int numThreads = 4)
{
    // Because this is random initialization we need to share it between the serial and GPU to ensure they are valid
    vector<Point3D> centroids = initializeCentroids(numCentroids, points);
    vector<Point3D> parallelCentroidCopy = centroids; // Copies the data, not the reference to ensure we are validating correctly https://www.geeksforgeeks.org/ways-copy-vector-c/
    string serialFilename = "serial-cpu.csv";
    string parallelFilename = "parallel-cpu.csv";
    // cout << "Performing Serial CPU" << endl;
    // performSerial(numEpochs, numCentroids, &centroids, points, serialFilename);
    cout << "\nPerforming Parallel CPU with " << numThreads << " threads" << endl;
    performParallel(numEpochs, &parallelCentroidCopy, points, parallelFilename, numThreads);
    // cout << "Files Equal: " << areFilesEqual(serialFilename, parallelFilename, true) << endl;
}

int main()
{
    // Read in the data
    cout << "Reading in Song Data" << endl;
    vector<Point3D> basePoints = readcsv("song_data.csv");
    // Run The Code with the same data
    vector<Point3D> points1 = basePoints;
    run(100, 6, &points1, 16);
    vector<Point3D> points2 = basePoints;
    run(100, 6, &points2, 24);
    vector<Point3D> points3 = basePoints;
    run(100, 6, &points3, 32);
}
