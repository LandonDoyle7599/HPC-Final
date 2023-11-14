#include "serial.cpp"
#include <iostream>

int main()
{
    // Read in the data
    cout << "Reading the csv" << endl;
    vector<Point3D> points = readcsv("song_data.csv");
    int numEpochs = 25;
    int numCentroids = 6;
    // Because this is random initialization we need to share it between the serial and GPU to ensure they are valid
    vector<Point3D> centroids = initializeCentroids(numCentroids, &points);
    string f1 = "serial-cpu.csv";
    cout << "Performing serial" << endl;
    performSerial(numEpochs, &centroids, &points, f1);
}