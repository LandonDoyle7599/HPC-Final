#include <mpi.h>

// Function to gather and update centroids across all processes
void gatherAndUpdateCentroids(vector<Point3D> *centroids, vector<Point3D> *localPoints)
{
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Gather all local centroids to the root process
    vector<Point3D> allCentroids(centroids->size() * size);
    MPI_Gather(centroids->data(), centroids->size() * sizeof(Point3D), MPI_BYTE,
               allCentroids.data(), centroids->size() * sizeof(Point3D), MPI_BYTE,
               0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Update global centroids based on the gathered information
        updateCentroidData(&allCentroids, centroids, centroids->size());
    }

    // Broadcast the updated centroids to all processes
    MPI_Bcast(centroids->data(), centroids->size() * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);
}

// Function to distribute points among processes
void distributePoints(vector<Point3D> *points, vector<Point3D> *localPoints)
{
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int pointsPerProcess = points->size() / size;
    int remainder = points->size() % size;

    int localSize = (rank < remainder) ? (pointsPerProcess + 1) : pointsPerProcess;
    int offset = rank * pointsPerProcess + min(rank, remainder);

    localPoints->resize(localSize);
    MPI_Scatter(points->data(), localSize * sizeof(Point3D), MPI_BYTE,
                localPoints->data(), localSize * sizeof(Point3D), MPI_BYTE,
                0, MPI_COMM_WORLD);
}

// Function to perform k-means clustering on a subset of points
void kMeansClusteringDistributedCPU(vector<Point3D> *localPoints, int numEpochs, vector<Point3D> *centroids)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int epoch = 0; epoch < numEpochs; ++epoch)
    {
        // Perform local k-means clustering on local points
        for (auto &point : *localPoints)
        {
            double minDist = numeric_limits<double>::max();
            int clusterId = -1;

            for (size_t i = 0; i < centroids->size(); ++i)
            {
                double dist = point.distance(centroids->at(i));
                if (dist < minDist)
                {
                    minDist = dist;
                    clusterId = static_cast<int>(i);
                }
            }

            point.minDist = minDist;
            point.cluster = clusterId;
        }

        // Gather information about local points to synchronize across processes
        vector<Point3D> allPoints(points.size() * size);
        MPI_Gather(localPoints->data(), localPoints->size() * sizeof(Point3D), MPI_BYTE,
                   allPoints.data(), localPoints->size() * sizeof(Point3D), MPI_BYTE,
                   0, MPI_COMM_WORLD);

        // Synchronize all points across processes
        MPI_Bcast(allPoints.data(), allPoints.size() * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Update centroids locally (use MPI_Reduce or similar for a global update)
        updateCentroidData(localPoints, centroids, centroids->size());

        // Gather and synchronize updated centroids
        MPI_Gather(centroids->data(), centroids->size() * sizeof(Point3D), MPI_BYTE,
                   allCentroids.data(), centroids->size() * sizeof(Point3D), MPI_BYTE,
                   0, MPI_COMM_WORLD);

        // Update centroids
        if (rank == 0)
        {
            updateCentroidData(localPoints, centroids, centroids->size());
        }

        // Share those updates with the other processes
        MPI_Bcast(centroids->data(), centroids->size() * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);
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
    distributePoints(points, &localPoints);

    // Perform k-means clustering on local points
    kMeansClusteringDistributedCPU(&localPoints, numEpochs, centroids);

    // Gather and update centroids across all processes
    gatherAndUpdateCentroids(centroids, &localPoints);

    // Print and save results in the root process
    if (rank == 0)
    {
        printStats(numEpochs, centroids->size(), points, 0);
        saveOutputs(points, filename);
    }
}
