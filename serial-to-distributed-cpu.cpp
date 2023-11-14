#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <mpi.h>
#include "serial.cpp"

using namespace std;

void calculateKMean(double k_x[], double k_y[], double k_z[],
                    double recv_x[], double recv_y[], double recv_z[], int assign[], int numElements, int numCentroids)
{
    for (int i = 0; i < numElements; ++i)
    {
        double min_dist = numeric_limits<double>::max();
        int clusterID = 0;
        for (int j = 0; j < numCentroids; ++j)
        // Find the closest centroid
        {
            double x = abs(recv_x[i] - k_x[j]);
            double y = abs(recv_y[i] - k_y[j]);
            double z = abs(recv_z[i] - k_z[j]);
            double temp_dist = (x * x) + (y * y) + (z * z);

            if (temp_dist < min_dist)
            {
                min_dist = temp_dist;
                clusterID = j;
            }
        }
        // Update the assignment
        assign[i] = clusterID;
    }
}

void updateCentroidDataDistributed(double k_means_x[], double k_means_y[], double k_means_z[],
                                   double data_x_points[], double data_y_points[], double data_z_points[], int k_assignment[], int numElements, int numCentroids)
{
    int numK = numCentroids;
    vector<int> nPoints(numK, 0);
    vector<double> sumX(numK, 0.0);
    vector<double> sumY(numK, 0.0);

    // Iterate over the centroids and compute the means for each value
    for (int i = 0; i < numElements; ++i)
    {
        int clusterID = k_assignment[i];
        nPoints[clusterID] += 1;
        sumX[clusterID] += data_x_points[i];
        sumY[clusterID] += data_y_points[i];
        // Reset the min distance is not needed, we don't use it in the distributed version. We take this into account when calculating the k mean
    }

    // Compute the new centroids
    for (int clusterId = 0; clusterId < numK; ++clusterId)
    {
        k_means_x[clusterId] = sumX[clusterId] / nPoints[clusterId];
        k_means_y[clusterId] = sumY[clusterId] / nPoints[clusterId];
    }
}

int main(int argc, char *argv[])
{

    MPI_Init(NULL, NULL);
    int numCentroids;
    int numEpochs;
    int numElements;
    int world_size;
    int world_rank;
    double startTime;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Setup the centroids (the final output)
    double *k_means_x = NULL;
    double *k_means_y = NULL;
    double *k_means_z = NULL;
    int *k_assignment = NULL; // the cluster assignment for each point put together

    // The data points
    double *data_x_points = NULL;
    double *data_y_points = NULL;
    double *data_z_points = NULL;

    // The received data points (local data)
    double *recv_x = NULL;
    double *recv_y = NULL;
    double *recv_z = NULL;
    int *recv_assign = NULL; // the cluster assignment for each point for the local data

    // Files to store
    string serialFilename = "serial.csv";
    string distFilename = "distributed.csv";

    if (world_rank == 0)
    {
        if (argc != 3)
        {
            cout << "Usage: " << argv[0] << " <numEpochs> <numCentroids>\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        numEpochs = atoi(argv[1]);
        numCentroids = atoi(argv[2]);

        MPI_Bcast(&numCentroids, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numEpochs, 1, MPI_INT, 0, MPI_COMM_WORLD);

        cout << "Reading input data from file...\n";
        vector<Point3D> pointData = readcsv("song_data.csv");

        // Now we initialize the centroids and perform serial implementation
        vector<Point3D> centeroids = initializeCentroids(numCentroids, &pointData);
        vector<Point3D> serialCentroidCopy = centeroids;
        vector<Point3D> serialPointCopy = pointData;
        performSerial(numEpochs, &serialCentroidCopy, &serialPointCopy, serialFilename);

        // Get numElements and brodcast for all processes
        numElements = pointData.size();
        MPI_Bcast(&numElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Allocate memory for the data points
        k_assignment = (int *)malloc(sizeof(int) * numElements);
        data_x_points = (double *)malloc(sizeof(double) * numElements);
        data_y_points = (double *)malloc(sizeof(double) * numElements);
        data_z_points = (double *)malloc(sizeof(double) * numElements);

        // add data from point data to the appropriate arrays

        // TODO: Check to see what pointData at 0 is. Could be bug here
        for (size_t i = 0; i < numElements; i++)
        {
            data_x_points[i] = pointData[i].x;
            data_y_points[i] = pointData[i].y;
            data_z_points[i] = pointData[i].z;
            k_assignment[i] = 0;
        }

        // Start the timer using MPI https://www.mcs.anl.gov/research/projects/mpi/tutorial/gropp/node139.html#:~:text=The%20elapsed%20(wall%2Dclock),n%22%2C%20t2%20%2D%20t1%20)%3B
        startTime = MPI_Wtime();
        // Setup the k_means vectors to proper sizes
        k_means_x = (double *)malloc(sizeof(double) * numCentroids);
        k_means_y = (double *)malloc(sizeof(double) * numCentroids);
        k_means_z = (double *)malloc(sizeof(double) * numCentroids);

        // TODO: Check to see what centroids at 0 is. Could be bug here
        for (int i = 0; i < numCentroids; i++)
        {
            k_means_x[i] = centeroids[i].x;
            k_means_y[i] = centeroids[i].y;
            k_means_z[i] = centeroids[i].z;
        }

        cout << "Running k-means algorithm for " << numEpochs << " iterations...\n";

        // Define the receiving vectors to be be big enough to hold the data. Add 1 in case there are extra data points
        recv_x = (double *)malloc(sizeof(double) * ((numElements / world_size) + 1));
        recv_y = (double *)malloc(sizeof(double) * ((numElements / world_size) + 1));
        recv_z = (double *)malloc(sizeof(double) * ((numElements / world_size) + 1));
        recv_assign = (double *)malloc(sizeof(double) * ((numElements / world_size) + 1));
    }
    else
    {
        // Receive the number of centroids and epochs
        MPI_Bcast(&numCentroids, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numEpochs, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Setup the k_means vectors to proper sizes
        k_means_x = (double *)malloc(sizeof(double) * numCentroids);
        k_means_y = (double *)malloc(sizeof(double) * numCentroids);
        k_means_z = (double *)malloc(sizeof(double) * numCentroids);

        // Setup the received vectors to proper sizes, accounting for any extra data points
        recv_x = (double *)malloc(sizeof(double) * ((numElements / world_size) + 1));
        recv_y = (double *)malloc(sizeof(double) * ((numElements / world_size) + 1));
        recv_z = (double *)malloc(sizeof(double) * ((numElements / world_size) + 1));
        recv_assign = (double *)malloc(sizeof(double) * ((numElements / world_size) + 1));
    }

    // Scatter data across processes but use displacements
    int send_counts[world_size];
    int displacements[world_size];

    // Break up the data into chunks accouting for evenly dividing the data
    for (int i = 0; i < world_size; ++i)
    {
        send_counts[i] = numElements / world_size;
        displacements[i] = i * send_counts[i];
    }
    // Add the remainder to each process and adjust the displacements
    for (int i = 0; i < numElements % world_size; ++i)
    {
        send_counts[i] += 1;
        displacements[i + 1] += 1;
    }

    if (world_rank == 0)
    {
        // Print the send counts and displacements
        cout << "Displacements will be : " << endl;
        for (int i = 0; i < world_size; ++i)
        {
            cout << displacements[i] << " ";
        }
        cout << endl;
        cout << "Send counts will be : " << endl;
        for (int i = 0; i < world_size; ++i)
        {
            cout << send_counts[i] << " ";
        }
        cout << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Scatterv for x points
    MPI_Scatterv(data_x_points, send_counts, displacements, MPI_DOUBLE,
                 recv_x, recv_x, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatterv for y points
    MPI_Scatterv(data_y_points, send_counts, displacements, MPI_DOUBLE,
                 recv_y, recv_y, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatterv for z points
    MPI_Scatterv(data_z_points, send_counts, displacements, MPI_DOUBLE,
                 recv_z, recv_z, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        cout << "Starting k-means algorithm for " << numEpochs << " iterations...\n";
    }
    for (int i = 0; i < numEpochs; ++i)
    {
        // Broadcast the centroids
        MPI_Bcast(k_means_x, numCentroids, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(k_means_y, numCentroids, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(k_means_z, numCentroids, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Scatter the assignments
        // Note: This assumes the number of centroids is evenly divisible by the number of processes
        MPI_Scatter(k_assignment, (numElements / world_size) + 1, MPI_INT,
                    recv_assign, (numElements / world_size) + 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate the new assignments
        calculateKMean(k_means_x, k_means_y, k_means_z, recv_x, recv_y, recv_z, recv_assign);

        // Gather the assignments
        MPI_Gather(recv_assign, (numElements / world_size) + 1, MPI_INT,
                   k_assignment, (numElements / world_size) + 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (world_rank == 0)
        {
            updateCentroidDataDistributed(k_means_x, k_means_y, k_means_z, data_x_points, data_y_points, data_z_points, k_assignment);
        }
    }

    // TODO: Figure out segmentation fault in this code
    if (world_rank == 0)
    {
        // Log the time to finish
        double finishTime = MPI_Wtime();
        long duration = (long)((finishTime - startTime) * 1000000);
        double v = finishTime - startTime;
        cout << "Time: " << v << " ms" << endl;
        // Convert the data back to a vector of points for saving
        vector<Point3D> computedPoints;
        computedPoints.reserve(numElements); // reserve space for the vector to avoid reallocation
        for (int i = 0; i < numElements; i++)
        {
            computedPoints.push_back(Point3D(data_x_points[i], data_y_points[i], data_z_points[i]));
        }
        // Now assign clusters to the points from the k_assign we already have
        for (int i = 0; i < numCentroids; i++)
        {
            computedPoints[i].cluster = k_assignment[i];
        }
        saveOutputs(&computedPoints, distFilename);
        printStats(numEpochs, numCentroids, &computedPoints, duration);
        areFilesEqual(serialFilename, distFilename, true);
    }
    MPI_Finalize();
}
