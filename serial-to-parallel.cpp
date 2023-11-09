#include "serial.cpp"
#include "parallel.cpp"
#include <iostream>
using namespace std;
int main()
{
    // Read in the data
    cout << "Reading in Song Data" << endl;
    vector<Point3D> points = readcsv("song_data.csv");
    int numEpochs = 100;
    int numCentroids = 6;
    // Because this is random initialization we need to share it between the serial and GPU to ensure they are valid
    vector<Point3D> centroids = initializeCentroids(numCentroids, &points);
    vector<Point3D> parallelCentroidCopy = centroids; // Copies the data, not the reference to ensure we are validating correctly https://www.geeksforgeeks.org/ways-copy-vector-c/
    string serialFilename = "serial-cpu.csv";
    string parallelFilename = "parallel-cpu.csv";
    cout << "Performing Serial CPU" << endl;
    performSerial(numEpochs, numCentroids, &centroids, &points, serialFilename);
    cout << "\nPerforming Parallel CPU" << endl;
    performParallel(numEpochs, &parallelCentroidCopy, &points, parallelFilename);
    cout << "Files Equal: " << areFilesEqual(serialFilename, parallelFilename, true) << endl;
}