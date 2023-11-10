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
        float minDist = points->at(i).distance(centroids->at(0));
        int clusterId = 0;
        for (int j = 1; j < numCentroids; ++j)
        {
            float dist = points->at(i).distance(centroids->at(j));
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
