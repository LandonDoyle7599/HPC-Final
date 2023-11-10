#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <ctime>
#include <limits>
#include <vector>
#include <mpi.h>
#include "serial.hpp"

using namespace std;

// Define the Point3D struct and other functions here...

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Verify correct number of arguments
    if (argc != 3)
    {
        if (rank == 0)
            cout << "Usage: " << argv[0] << " <numEpochs> <numCentroids>\n";
        MPI_Finalize();
        return 1;
    }
    string filename = "distributed-cpu.csv";
    string serialFilename = "serial.csv";
    int numEpochs = stoi(argv[2]);
    int numCentroids = stoi(argv[3]);
    vector<Point3D> points;

    // Rank 0 reads in the file and initializes the centroids
    if (rank == 0)
    {
        // Read data on the root process
        points = readcsv("song_data.csv");
        // Initialize centroids on the root process
        vector<Point3D> centroids = initializeCentroids(numCentroids, &points);

        // Make copies and perfrom serial for later comparison
        vector<Point3D> serialPoints = points;
        vector<Point3D> serialCentroids = centroids;
        performSerial(numEpochs, &serialCentroids, &serialPoints, serialFilename);

        // Perform parallel k-means clustering using MPI
        performDistributed(numEpochs, &centroids, &points, filename);
    }
    else
    {
        // Other processes only participate in MPI communications
        performDistributed(numEpochs, nullptr, nullptr, filename);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        // Now we validate the outputs
        cout << "Validating outputs..." << endl;
        areFilesEqual(serialFilename, filename, true)
    }

    MPI_Finalize();
    return 0;
}
