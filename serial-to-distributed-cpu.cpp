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

    // Broadcast centroid and epoch data to all processes
    MPI_Bcast(centroids.data(), centroids.size() * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numCentroids, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numEpochs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Distribute points among processes and compensate for uneven division
    vector<Point3D> localPoints;
    int pointsPerProcess = points.size() / size;
    int remainder = points.size() % size;
    int localSize = (rank < remainder) ? (pointsPerProcess + 1) : pointsPerProcess;
    int offset = rank * pointsPerProcess + min(rank, remainder);
    localPoints.resize(localSize);

    // Distribute the points to all processes
    MPI_Scatter(points.data(), localSize * sizeof(Point3D), MPI_BYTE,
                localPoints.data(), localSize * sizeof(Point3D), MPI_BYTE,
                0, MPI_COMM_WORLD);

    cout << "Number of processes: " << size << endl;
    cout << "Performing Distributed CPU from rank: " << rank << endl;

    // Perform k-means clustering on local points
    // Repeat over epochs to converge the centroids
    for (int epoch = 0; epoch < numEpochs; ++epoch)
    {
        cout << " Rank: " << rank << " Epoch: " << epoch << " of " << numEpochs << endl;
        // For the given processor's localPoints, compute the nearest centroid to it
        kMeansClusteringCPU(&localPoints, &centroids, localPoints.size(), centroids.size());
        cout << " Rank: " << rank << " Completed Clustering " << endl;

        // Update the centroids
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
        {
            cout << "Updating Centroids from Rank 0" << endl;
            updateDistributedCentroidData(*localPoints, *centroids, centroids.size());
        }
    }

    // Print and save results in the root process
    if (rank == 0)
    {
        cout << "Saving outputs..." << endl;
        printStats(numEpochs, centroids.size(), points, 0);
        saveOutputs(points, filename);
    }

    // Wrap everything up
    if (rank == 0)
    {
        // Now we validate the outputs
        cout << "Validating outputs..." << endl;
        areFilesEqual(serialFilename, filename, true);
    }

    MPI_Finalize();
    return 0;
}
