#include <iostream>
#include "serial.cpp"
#include <vector>
#include <mpi.h>

using namespace std;

void updateCentroidDataMPI(vector<Point3D> &localPoints, vector<Point3D> &centroids, int numCentroids, int rank)
{
    // Create vectors to keep track of data needed to compute means
    vector<int> nPoints(numCentroids, 0);
    vector<double> sumX(numCentroids, 0.0);
    vector<double> sumY(numCentroids, 0.0);

    // Iterate over points to append data to centroids
    for (vector<Point3D>::iterator it = localPoints->begin(); it != localPoints->end(); ++it)
    {
        int clusterId = it->cluster;
        nPoints[clusterId] += 1;
        sumX[clusterId] += it->x;
        sumY[clusterId] += it->y;

        it->minDist = numeric_limits<float>::max(); // reset distance
    }

    if (rank == 0)
    {
        // Compute the new centroids
        for (vector<Point3D>::iterator c = centroids->begin(); c != centroids->end(); ++c)
        {
            int clusterId = c - centroids->begin();
            c->x = sumX[clusterId] / nPoints[clusterId];
            c->y = sumY[clusterId] / nPoints[clusterId];
        }
    }
    // Broadcast the updated centroids to all processes
    MPI_Bcast(centroids.data(), centroids.size() * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);
}

void kMeansClusteringParallelMPI(vector<Point3D> &points, int numEpochs, vector<Point3D> &centroids, int rank, int size)
{

    cout << "Rank: " << rank << " is part of kMeansClustering " << endl;
    int localSize = points->size() / size;
    int localStart = rank * localSize;
    int localEnd = (rank == size - 1) ? points->size() : localStart + localSize;

    for (int epoch = 0; epoch < numEpochs; ++epoch)
    {
        cout << "Rank: " << rank << " on epoch: " << epoch << endl;

        for (int i = localStart; i < localEnd; ++i)
        {
            Point3D &p = points->at(i);
            int clusterId = 0;
            double minDist = centroids[0].distance(p);

            for (int j = 1; j < centroids.size(); ++j)
            {
                double dist = centroids[j].distance(p);
                if (dist < minDist)
                {
                    minDist = dist;
                    clusterId = j;
                }
            }

            p.minDist = minDist;
            p.cluster = clusterId;
        }

        cout << "Rank: " << rank << " Completed epoch: " << epoch << endl;
        // Perform a global reduction to update centroids
        updateCentroidDataMPI(points, centroids, centroids.size(), rank);
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    // Get rank and get size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    vector<Point3D> basePoints;
    vector<Point3D> centroids;
    int numEpochs = 25;
    int numClusters = 6;
    string serialFilename = "serial-cpu.csv";
    string distributedFilename = "distributed-cpu.csv";
    // Read in the data on rank 0

    if (rank == 0)
    {
        cout << "Reading in Song Data" << endl;
        basePoints = readcsv("song_data.csv");
        centroids = initializeCentroids(6, &basePoints);
        vector<Point3D> serialCentroidCopy = centroids;
        vector<Point3D> serialPointsCopy = basePoints;
        performSerial(numEpochs, &serialCentroidCopy, &serialPointsCopy, serialFilename);
        cout << "Performing Distributed CPU" << endl;
    }

    // Broadcast the size of the data to all ranks
    int dataSize = basePoints.size();
    MPI_Bcast(&dataSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter the data among all ranks
    vector<Point3D> localPoints(dataSize / size);
    MPI_Scatter(basePoints.data(), localPoints.size() * sizeof(Point3D), MPI_BYTE,
                localPoints.data(), localPoints.size() * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Broadcast centroids to all ranks
    MPI_Bcast(centroids.data(), centroids.size() * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);

    cout << "Rank " << rank << " has " << localPoints.size() << " points" << endl;
    // Execute k-means clustering
    kMeansClusteringParallelMPI(localPoints, numEpochs, centroids, rank, size);

    // Now that it is finished executing, gather the data back toegerh to review
    if (rank == 0)
    {
        printStats(numEpochs, centroids.size(), &basePoints, 0);
        saveOutputs(&basePoints, distributedFilename);
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
