#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <ctime>
#include <limits>
#include <vector>
#include <mpi.h>
#include "serial.cpp"
#include "distributedCPU.cpp"

using namespace std;

// Define the Point3D struct and other functions here...

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Verify correct number of arguments
    if (rank == 0)
    {
        if (argc != 3)
        {
            cout << "Usage: " << argv[0] << " <numEpochs> <numCentroids>\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    string filename = "distributed-cpu.csv";
    string serialFilename = "serial.csv";
    int numEpochs = atoi(argv[1]);
    int numCentroids = atoi(argv[2]);
    vector<Point3D> points;
    vector<Point3D> centroids;
    // Rank 0 reads in the file and initializes the centroids
    if (rank == 0)
    {
        cout << "Reading Data " << endl;
        // Read data on the root process
        points = readcsv("song_data.csv");
        // Initialize centroids on the root process
        centroids = initializeCentroids(numCentroids, &points);

        // Make copies and perfrom serial for later comparison
        vector<Point3D> serialPoints = points;
        vector<Point3D> serialCentroids = centroids;
        performSerial(numEpochs, &serialCentroids, &serialPoints, serialFilename);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // Perform parallel k-means clustering using MPI
    performDistributed(numEpochs, &centroids, &points, filename);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        // Now we validate the outputs
        cout << "Validating outputs..." << endl;
        areFilesEqual(serialFilename, filename, true);
    }

    MPI_Finalize();
    return 0;
}
