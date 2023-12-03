
#include "serial.cpp"
#include "gpu.cu"
#include <iostream>
using namespace std;
int main(int argc, char **argv)
{
    int numEpochs;
    int numCentroids;
    getEpochsCentroids(argc, argv, numEpochs, numCentroids);
    // Read in the data
    cout << "Reading the csv" << endl;
    vector<Point3D> points = readcsv("song_data.csv");
    // Because this is random initialization we need to share it between the serial and GPU to ensure they are valid
    vector<Point3D> centroids = initializeCentroids(numCentroids, &points);
    vector<Point3D> gpuCentroidCopy = centroids; // Copies the data, not the reference to ensure we are validating correctly https://www.geeksforgeeks.org/ways-copy-vector-c/
    vector<Point3D> gpuPointsCopy = points;
    string f1 = "serial-cpu.csv";
    string f2 = "gpu.csv";
    performSerial(numEpochs, &centroids, &points, f1);
    cout << "\nPerforming GPU" << endl;
    performGPU(numEpochs, &gpuCentroidCopy, &gpuPointsCopy, f2);
    areFilesEqual(f1, f2, true);
}