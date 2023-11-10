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
    MPI_Datatype mpi_point_type = createPoint3DType();
    string filename = "distributed-cpu.csv";
    string serialFilename = "serial.csv";
    vector<Point3D> points;
    vector<Point3D> centroids;
    vector<Point3D> localPoints;
    int numPoints;
    int numCentroids;
    int numEpochs;

    if (rank == 0)
    {
        // Read input
        cout << "Number of processes: " << size << endl;
        if (argc != 3)
        {
            cout << "Usage: " << argv[0] << " <numEpochs> <numCentroids>\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        numEpochs = atoi(argv[1]);
        numCentroids = atoi(argv[2]);
        MPI_Bcast(&numEpochs, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numCentroids, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Read data on the root process
        cout << "Reading Data " << endl;
        points = readcsv("song_data.csv");
        numPoints = points.size();
        MPI_Bcast(&numPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Initialize centroids on the root process
        centroids = initializeCentroids(numCentroids, &points);
        // Make copies and perfrom serial for later comparison
        vector<Point3D> serialPoints = points;
        vector<Point3D> serialCentroids = centroids;
        performSerial(numEpochs, &serialCentroids, &serialPoints, serialFilename);
    }
    else
    {
        // Receive the number of epochs, centroids, and numPoints from the root process
        MPI_Bcast(&numEpochs, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numCentroids, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numPoints, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Distribute points among processes and compensate for uneven division
    int pointsPerProcess = points.size() / size;
    int remainder = points.size() % size;
    int localSize = (rank < remainder) ? (pointsPerProcess + 1) : pointsPerProcess;
    int offset = rank * pointsPerProcess + min(rank, remainder);
    localPoints.resize(localSize);

    // Distribute the points to all processes
    MPI_Scatter(points.data(), localSize, mpi_point_type,
                localPoints.data(), localSize, mpi_point_type,
                0, MPI_COMM_WORLD);

    cout << "Performing Distributed CPU from rank: " << rank << endl;

    // Perform k-means clustering on local points
    // Repeat over epochs to converge the centroids
    for (int epoch = 0; epoch < numEpochs; ++epoch)
    {
        cout << "Rank: " << rank << " Epoch: " << epoch << endl;
        // get the local centroids
        MPI_Bcast(centroids.data(), centroids.size(), mpi_point_type, 0, MPI_COMM_WORLD);
        // Scatter the points to all processes
        MPI_Scatter(points.data(), localSize, mpi_point_type,
                    localPoints.data(), localSize, mpi_point_type,
                    0, MPI_COMM_WORLD);

        kMeansClusteringCPU(&localPoints, &centroids, localPoints.size(), centroids.size());

        // Now gather the local points from all processes to the root process
        vector<Point3D> allLocalPoints(numPoints);
        MPI_Gather(localPoints.data(), localPoints.size() * sizeof(Point3D), MPI_BYTE,
                   allLocalPoints.data(), localPoints.size() * sizeof(Point3D), MPI_BYTE,
                   0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
        {
            cout << "Updating Centroids from Rank 0" << endl;
            // Update global centroids based on the gathered information
            // We need all points and all centroids updated in order to properly update
            updateCentroidData(&allLocalPoints, &centroids, numCentroids);
        }
    }

    // Print and save results in the root process
    if (rank == 0)
    {
        cout << "Saving outputs..." << endl;
        printStats(numEpochs, centroids.size(), &points, 0);
        saveOutputs(&points, filename);
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
