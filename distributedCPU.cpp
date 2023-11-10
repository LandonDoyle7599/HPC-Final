#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>
#include <vector>
#include <mpi.h>

using namespace std;

/**
 * Perform k-means clustering using OpenMPI
 * @param points - pointer to vector of points
 * @param numEpochs - number of k-means iterations
 * @param centroids - pointer to vector of centroids
 */
void kMeansClusteringDistributedCPU(vector<Point3D> *points, int numEpochs, vector<Point3D> *centroids)
{
    MPI_Init();
    int rank, size;
    int numPoints = points->size();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int pointsPerProcess = numPoints / size;
    // Need to assign points to each process
    int startPoint = rank * pointsPerProcess;
    int endPoint = (rank == size - 1) ? numPoints : startPoint + pointsPerProcess;

    for (int epoch = 0; epoch < numEpochs; ++epoch)
    {
        // Broadcast centroids to all processes. We need to update each node of the current centroids for each epoch to ensure we are using the most up to date data
        MPI_Bcast(&centroids->front(), centroids->size() * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Each process will compute the distance for its given set of points
        for (int i = startPoint; i < endPoint; ++i)
        {
            Point3D &p = (*points)[i];
            int clusterId = 0;
            double minDist = centroids->at(0).distance(p);

            for (int j = 1; j < centroids->size(); ++j)
            {
                double dist = centroids->at(j).distance(p);
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
        MPI_Gather(&points->at(startPoint), pointsPerProcess * sizeof(Point3D), MPI_BYTE, &points->front(), pointsPerProcess * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Update the centroids on the root process with the collected data
        if (rank == 0)
        {
            updateCentroidData(points, centroids, centroids->size());
        }
    }
    MPI_Finalize();
}

void performDistributedCPU(int numEpochs, vector<Point3D> *centroids, vector<Point3D> *points, string filename)
{
    // Time code: https://stackoverflow.com/questions/21856025/getting-an-accurate-execution-time-in-c-micro-seconds
    cout << "Entering the k means computation" << endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    kMeansClusteringDistributedCPU(points, numEpochs, centroids); // K-means clustering on the points.
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    printStats(numEpochs, centroids->size(), points, duration.count());
    saveOutputs(points, filename);
}
