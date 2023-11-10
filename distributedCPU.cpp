#include <mpi.h>

// Function to update distributed centroids based on local information
void updateDistributedCentroidData(vector<Point3D> &localPoints, vector<Point3D> &centroids, int numCentroids)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Gather local centroids from all processes to the root process
    vector<Point3D> allCentroids(centroids.size() * size);
    MPI_Gather(centroids.data(), centroids.size() * sizeof(Point3D), MPI_BYTE,
               allCentroids.data(), centroids.size() * sizeof(Point3D), MPI_BYTE,
               0, MPI_COMM_WORLD);

    // Gather local points from all processes to the root process
    vector<Point3D> allPoints(localPoints.size() * size);
    MPI_Gather(localPoints.data(), localPoints.size() * sizeof(Point3D), MPI_BYTE,
               allPoints.data(), localPoints.size() * sizeof(Point3D), MPI_BYTE,
               0, MPI_COMM_WORLD);

    // Update global centroids based on the gathered information
    updateCentroidData(&allPoints, &allCentroids, numCentroids);

    // Broadcast the updated centroids to all processes
    MPI_Bcast(centroids.data(), centroids.size() * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);
}

void kMeansClusteringCPU(vector<Point3D> *points, vector<Point3D> *centroids, int nPoints, int numCentroids)
{
    for (int i = 0; i < nPoints; ++i)
    {
        // Initialize to first value
        float minDist = points->at(i).distance(centroids[0]);
        int clusterId = 0;
        for (int j = 1; j < numCentroids; ++j)
        {
            float dist = points->at(i).distance(centroids[j]);
            if (dist < minDist)
            {
                minDist = dist;
                clusterId = j;
            }
        }
        points->at(i).minDist = minDist;
        points->at(i).cluster = clusterId;
    }
}

// Function to perform k-means clustering on a subset of points
void kMeansClusteringMPI(vector<Point3D> *points, int numEpochs, vector<Point3D> *centroids)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Repeat over epochs to converge the centroids
    for (int epoch = 0; epoch < numEpochs; ++epoch)
    {
        // For each centroid, compute distance from centroid to each point
        // and update point's cluster if necessary
        kMeansClusteringCPU(points, centroids, points->size(), centroids->size());

        // Update the centroids
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
        {
            updateDistributedCentroidData(*points, *centroids, centroids->size());
        }
    }
}

// Main function for parallel k-means clustering using MPI
void performDistributed(int numEpochs, vector<Point3D> *centroids, vector<Point3D> *points, string filename)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Distribute points among processes
    vector<Point3D> localPoints;

    int pointsPerProcess = points->size() / size;
    int remainder = points->size() % size;

    int localSize = (rank < remainder) ? (pointsPerProcess + 1) : pointsPerProcess;
    int offset = rank * pointsPerProcess + min(rank, remainder);

    localPoints.resize(localSize);
    MPI_Scatter(points->data(), localSize * sizeof(Point3D), MPI_BYTE,
                localPoints->data(), localSize * sizeof(Point3D), MPI_BYTE,
                0, MPI_COMM_WORLD);

    // Broadcast centroids to all processes
    MPI_Bcast(centroids->data(), centroids->size() * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);

    cout << "Performing Distributed CPU from rank: " << rank << endl;

    // Perform k-means clustering on local points
    kMeansClusteringMPI(&localPoints, numEpochs, centroids);

    // Gather and update centroids across all processes
    // gatherAndUpdateCentroids(centroids, &localPoints);

    // Print and save results in the root process
    if (rank == 0)
    {
        cout << "Saving outputs..." << endl;
        printStats(numEpochs, centroids->size(), points, 0);
        saveOutputs(points, filename);
    }
}
