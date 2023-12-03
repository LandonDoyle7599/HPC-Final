#include "serial.cpp"
#include "parallel.cpp"
#include <iostream>
using namespace std;

void run(int numEpochs, int numCentroids, vector<Point3D> *points, int numThreads)
{
    // Because this is random initialization we need to share it between the serial and GPU to ensure they are valid
    vector<Point3D> centroids = initializeCentroids(numCentroids, points);
    // Copies the data, not the reference to ensure we are validating correctly https://www.geeksforgeeks.org/ways-copy-vector-c/
    vector<Point3D> parallelCentroidCopy = centroids;
    vector<Point3D> paralellPointsCopy = *points;
    string serialFilename = "serial-cpu.csv";
    string parallelFilename = "parallel-cpu.csv";
    // Execute operations
    performSerial(numEpochs, &centroids, points, serialFilename);
    cout << "\nPerforming Parallel CPU with " << numThreads << " threads" << endl;
    performParallel(numEpochs, &parallelCentroidCopy, &paralellPointsCopy, parallelFilename, numThreads);
    areFilesEqual(serialFilename, parallelFilename, true);
}

int main(int argc, char **argv)
{
    int numEpochs;
    int numCentroids;
    int numThreads;
    getEpochsCentroids(argc, argv, numEpochs, numCentroids);
    if (argc < 4)
    {
        cout << "No user entered thread count, defaulting to 4" << endl;
        numThreads = 4;
    }
    else
    {
        numThreads = atoi(argv[3]);
    }
    // Read in the data
    cout << "Reading in Song Data" << endl;
    vector<Point3D> basePoints = readcsv("song_data.csv");
    // Run The Code with the same data
    vector<Point3D> points1 = basePoints;
    run(numEpochs, numCentroids, &points1, numThreads);
}
