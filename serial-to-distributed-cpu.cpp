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
    int numCentroids;
    int numEpochs;
    int numPoints;

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
        // Read data on the root process
        cout << "Reading Data " << endl;
        points = readcsv("song_data.csv");
        numPoints = points.size();
        // Initialize centroids on the root process
        centroids = initializeCentroids(numCentroids, &points);
        // Make copies and perfrom serial for later comparison
        vector<Point3D> serialPoints = points;
        vector<Point3D> serialCentroids = centroids;
        performSerial(numEpochs, &serialCentroids, &serialPoints, serialFilename);
    }

    // Broadcast centroid and epoch data to all processes from thread 0
    MPI_Bcast(centroids.data(), centroids.size(), mpi_point_type, 0, MPI_COMM_WORLD);
    cout << "Broadcasted centroids " << rank << endl;
    MPI_Bcast(&numCentroids, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cout << "Broadcasted numCentroids " << rank << endl;
    MPI_Bcast(&numEpochs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cout << "Broadcasted numEpochs  " << rank << endl;

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
        MPI_Barrier(MPI_COMM_WORLD);
        cout << " Rank: " << rank << " Epoch: " << epoch << " of " << numEpochs << endl;
        // For the given processor's localPoints, compute the nearest centroid to it
        kMeansClusteringCPU(&localPoints, &centroids, localPoints.size(), centroids.size());
        cout << " Rank: " << rank << " Completed Clustering " << endl;
        if (rank == 0)
        {
            cout << "Updating Centroids from Rank 0" << endl;
            // Gather local centroids from all processes to the root process
            // vector<Point3D> allCentroids(numCentroids);
            // MPI_Gather(centroids.data(), centroids.size() * sizeof(Point3D), MPI_BYTE,
            //            allCentroids.data(), centroids.size() * sizeof(Point3D), MPI_BYTE,
            //            0, MPI_COMM_WORLD);

            // // Gather local points from all processes to the root process
            // vector<Point3D> allPoints(numPoints);
            // MPI_Gather(localPoints.data(), localPoints.size() * sizeof(Point3D), MPI_BYTE,
            //            allPoints.data(), localPoints.size() * sizeof(Point3D), MPI_BYTE,
            //            0, MPI_COMM_WORLD);

            // Update global centroids based on the gathered information
            // We need all points and all centroids updated in order to properly update
            // updateCentroidData(&allPoints, &allCentroids, numCentroids);
            updateCentroidData(&points, &centroids, numCentroids);

            // Broadcast the updated centroids to all processes
            // MPI_Bcast(centroids.data(), centroids.size(), mpi_point_type, 0, MPI_COMM_WORLD);
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
