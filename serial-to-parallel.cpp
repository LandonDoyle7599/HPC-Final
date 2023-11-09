
#include "serial.cpp"
#include "parallel.cpp"
#include <iostream>
using namespace std;
int main()
{
    // Read in the data
    cout << "Reading the csv" << endl;
    vector<Point3D> points = readcsv("song_data.csv");
    int numEpochs = 100;
    int numCentroids = 6;
    // Because this is random initialization we need to share it between the serial and GPU to ensure they are valid
    vector<Point3D> centroids = initializeCentroids(numCentroids, &points);
    vector<Point3D> parallelCentroidCopy = centroids; // Copies the data, not the reference to ensure we are validating correctly https://www.geeksforgeeks.org/ways-copy-vector-c/
    string f1 = "serial-cpu.csv";
    string f2 = "gpu.csv";
    cout << "Performing serial" << endl;
    performSerial(numEpochs, numCentroids, &centroids, &points, f1);
    cout << "\nPerforming GPU" << endl;
    performParallel(numEpochs, numCentroids, &parallelCentroidCopy, &points, f2);
    cout << "Files Equal: " << areFilesEqual(f1, f2) << endl;
}