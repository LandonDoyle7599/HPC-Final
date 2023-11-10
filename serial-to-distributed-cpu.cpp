#include "serial.cpp"
#include "distributedCPU.cpp"
#include <iostream>
using namespace std;

void run(int numEpochs, int numCentroids, vector<Point3D> *points)
{
    // Because this is random initialization we need to share it between the serial and distributed implementations
    vector<Point3D> centroids = initializeCentroids(numCentroids, points);
    // Copies the data to ensure we are validating correctly
    // https://www.geeksforgeeks.org/ways-copy-vector-c/
    vector<Point3D> distributedCentroidCopy = centroids;
    vector<Point3D> distributedPointsCopy = *points;
    string serialFilename = "serial-cpu.csv";
    string distributedFilename = "distributed-cpu.csv";
    // Execute the operations
    cout << "Performing Serial CPU" << endl;
    performSerial(numEpochs, &centroids, points, serialFilename);
    cout << "\nPerforming Distributed CPU " << endl;
    performDistributedCPU(numEpochs, &distributedCentroidCopy, &distributedPointsCopy, distributedFilename);

    // Compare outputs to validate they computed the same values
    bool debug = true;
    if (debug)
    {
        areFilesEqual(serialFilename, distributedFilename, debug);
    }
    else
    {
        cout << "Files Equal: " << areFilesEqual(serialFilename, distributedFilename, debug) << endl;
    }
}

int main()
{
    // Read in the data
    cout << "Reading in Song Data" << endl;
    vector<Point3D> basePoints = readcsv("song_data.csv");
    // Run The Code with a copy of the song data
    vector<Point3D> points1 = basePoints;
    run(25, 6, &points1);
}
