#include "serial.cpp"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>
#include <vector>
#include <mpi.h>
using namespace std;

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    // Get rank and get size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Initialize variables
    int numEpochs;
    int numPoints;
    vector<Point3D> centroids;
    vector<Point3D> points;
    string serialFilename = "serial-cpu.csv";
    string distributedFilename = "distributed-cpu.csv";

    // Rank 0 executes the serial code
    if (rank == 0)
    {
        // Read in data from the command line
        if (argc != 3)
        {
            cout << "Usage: " << argv[0] << " <numEpochs> <numCentroids>" << endl;
            return 1;
        }
        numEpochs = atoi(argv[1]);
        int numCentroids = atoi(argv[2]);

        cout << "Number of processes: " << size << endl;
        // Read in the data
        cout << "Reading in Song Data" << endl;
        points = readcsv("song_data.csv");
        numPoints = points.size();
        // Run serial code with a copy of the song data
        vector<Point3D> serialPoints = points;
        // Because this is random initialization we need to share it between the serial and distributed implementations
        centroids = initializeCentroids(numCentroids, &points);
        // Copies the data to ensure we are validating correctly https://www.geeksforgeeks.org/ways-copy-vector-c/
        vector<Point3D> serialCentroidCopy = centroids;
        // Execute the operations
        cout << "Performing Serial CPU" << endl;
        performSerial(numEpochs, &serialCentroidCopy, &serialPoints, serialFilename);
    }
    // Wait until the serial code is done before starting the distributed computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = std::chrono::high_resolution_clock::now();
    if (rank == 0)
    {
        cout << "Performing Distributed CPU" << endl;
    }
    // Define how much each process will work on each epoch

    int pointsPerProcess = numPoints / size;
    int startPoint = rank * pointsPerProcess;
    //  If 0 on the first iteration, then add the previous send count to the displacement
    int endPoint = (rank == size - 1) ? numPoints : startPoint + pointsPerProcess;
    vector<Point3D> localPoints(pointsPerProcess);

    // Buffer to receive the local points for each node
    for (int epoch = 0; epoch < numEpochs; ++epoch)
    {
        if (rank == 0)
        {
            cout << "Epoch number: " << epoch << endl;
        }
        // Broadcast centroids to all processes. We need to update each node of the current centroids for each epoch to ensure we are using the most up to date data and actually converging.
        MPI_Bcast(centroids.data(), centroids.size() * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Scatter the points to each process so they can work on them. Once again, we need to update the data each epoch in order to converge
        MPI_Scatter(points.data(), pointsPerProcess * sizeof(Point3D), MPI_BYTE,
                    localPoints.data(), pointsPerProcess * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Each process will compute the distance for its given set of points.
        for (int i = 0; i < pointsPerProcess; ++i)
        {
            Point3D &p = localPoints[i];
            int clusterId = 0;
            double minDist = centroids.at(0).distance(p);
            for (int j = 1; j < centroids.size(); ++j)
            {
                double dist = centroids.at(j).distance(p);
                if (dist < minDist)
                {
                    minDist = dist;
                    clusterId = j;
                }
            }
            p.minDist = minDist;
            p.cluster = clusterId;
        }

        // Gather updated points to the root process
        MPI_Gather(localPoints.data(), pointsPerProcess * sizeof(Point3D), MPI_BYTE,
                   points.data(), pointsPerProcess * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Update the centroids on the root process with the collected data
        if (rank == 0)
        {
            updateCentroidData(&points, &centroids, centroids.size());
        }
    }

    // Now we simply compare the outputs
    // Only want 1 process to compare the files
    if (rank == 0)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        printStats(numEpochs, centroids.size(), &points, duration.count());
        saveOutputs(&points, distributedFilename);
        // Compare outputs to validate they computed the same values
        bool debug = true;
        if (debug)
        {
            areFilesEqual(serialFilename, distributedFilename, debug);
        }
        else
        {
            cout << "Files Equal: " << areFilesEqual(serialFilename, distributedFilename, debug) << endl;
        }
    }
    MPI_Finalize();

    return 0;
}
