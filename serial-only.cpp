#include "serial.cpp"
#include <iostream>

int main(int argc, char **argv)
{
    // Read in the data
    cout << "Reading the csv" << endl;
    vector<Point3D> points = readcsv("song_data.csv");
    // Get the epochs and centroids from the command line
    int numEpochs;
    int numCentroids;
    getEpochsCentroids(argc, argv, numEpochs, numCentroids);
    // Because this is random initialization we need to share it between the serial and GPU to ensure they are valid
    vector<Point3D> centroids = initializeCentroids(numCentroids, &points);
    string f1 = "serial-cpu.csv";
    performSerial(numEpochs, &centroids, &points, f1);
}