#include "serial.cpp"
#include "distributedCPU.cpp"
#include <iostream>
using namespace std;

void run(int numEpochs, int numCentroids, vector<Point3D> *points)
{
    // Because this is random initialization we need to share it between the serial and GPU to ensure they are valid
    vector<Point3D> centroids = initializeCentroids(numCentroids, points);
    // Copies the data to ensure we are validating correctly
    // https://www.geeksforgeeks.org/ways-copy-vector-c/
    vector<Point3D> distributedCentroidCopy = centroids;
    vector<Point3D> distributedPointsCopy = *points;
    string serialFilename = "serial-cpu.csv";
    string distributedFilename = "distributed-cpu.csv";
    // Execute the operations
    cout << "Performing Serial CPU" << endl;
    performSerial(numEpochs, numCentroids, &centroids, points, serialFilename);
    cout << "\nPerforming Distributed CPU " << endl;
    performDistributedCPU(numEpochs, &centroids, &distributedPointsCopy, distributedFilename);
    // Compare outputs to validate they computed the same values
    cout << "Files Equal: " << areFilesEqual(serialFilename, distributedFilename, true) << endl;
}

int main()
{
    // Read in the data
    cout << "Reading in Song Data" << endl;
    vector<Point3D> basePoints = readcsv("song_data.csv");
    // Run The Code with the same data
    vector<Point3D> points1 = basePoints;
    run(100, 6, &points1);
    // vector<Point3D> points2 = basePoints;
    // run(200, 6, &points2, 12);
    // vector<Point3D> points3 = basePoints;
    // run(400, 6, &points3, 12);
    // vector<Point3D> points4 = basePoints;
    // run(800, 6, &points4, 12);
}
