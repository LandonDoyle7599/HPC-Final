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
                    double recv_x[], double recv_y[], double recv_z[], int assign[], int numLocalDataPoints, int numCentroids)
{
    for (int i = 0; i < numLocalDataPoints; ++i)
    {
        // Initialize min dist on the first centroid
        double min_dist = calculateDistanceSerial(recv_x[i], recv_y[i], recv_z[i], k_x[0], k_y[0], k_z[0]);
        int clusterID = 0;
        // Now find closest centroid to that point by looking at distance to all of them from this point
        for (int j = 1; j < numCentroids; ++j)
        {
            double temp_dist = calculateDistanceSerial(recv_x[i], recv_y[i], recv_z[i], k_x[j], k_y[j], k_z[j]);

            if (temp_dist < min_dist)
            {
                min_dist = temp_dist;
                clusterID = j;
            }
        }
        // Update the assignment of closest centroid to this point
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
    vector<double> sumZ(numK, 0.0);

    // Iterate over the centroids and compute the means for each value
    for (int i = 0; i < numElements; ++i)
    {
        int clusterID = k_assignment[i];
        nPoints[clusterID] += 1;
        sumX[clusterID] += data_x_points[i];
        sumY[clusterID] += data_y_points[i];
        sumZ[clusterID] += data_z_points[i];
        // Reset the min distance is not needed, we don't use it in the distributed version. We take this into account when calculating the k mean
    }

    // Compute the new centroids
    for (int clusterId = 0; clusterId < numK; ++clusterId)
    {
        k_means_x[clusterId] = sumX[clusterId] / nPoints[clusterId];
        k_means_y[clusterId] = sumY[clusterId] / nPoints[clusterId];
        k_means_z[clusterId] = sumZ[clusterId] / nPoints[clusterId];
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
        // Read command line args
        getEpochsCentroids(argc, argv, numEpochs, numCentroids);
        // Share their values with the other processes
        MPI_Bcast(&numCentroids, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numEpochs, 1, MPI_INT, 0, MPI_COMM_WORLD);

        cout << "Reading input data from file...\n";
        vector<Point3D> pointData = readcsv("song_data.csv");

        // Now we initialize the centroids and perform serial implementation
        vector<Point3D> centeroids = initializeCentroids(numCentroids, &pointData);
        vector<Point3D> serialCentroidCopy = centeroids;
        vector<Point3D> serialPointCopy = pointData;
        performSerial(numEpochs, &serialCentroidCopy, &serialPointCopy, serialFilename);

        // Start the timer using MPI https://www.mcs.anl.gov/research/projects/mpi/tutorial/gropp/node139.html#:~:text=The%20elapsed%20(wall%2Dclock),n%22%2C%20t2%20%2D%20t1%20)%3B
        cout << "Performing Distributed CPU" << endl;
        startTime = MPI_Wtime();

        // Get numElements and brodcast for all processes
        numElements = pointData.size();
        MPI_Bcast(&numElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Allocate memory for the data points
        k_assignment = (int *)malloc(sizeof(int) * numElements);
        data_x_points = (double *)malloc(sizeof(double) * numElements);
        data_y_points = (double *)malloc(sizeof(double) * numElements);
        data_z_points = (double *)malloc(sizeof(double) * numElements);

        // add data from point data to the appropriate arrays
        for (size_t i = 0; i < numElements; i++)
        {
            data_x_points[i] = pointData[i].x;
            data_y_points[i] = pointData[i].y;
            data_z_points[i] = pointData[i].z;
            k_assignment[i] = 0;
        }

        // Setup the k_means vectors to proper sizes
        k_means_x = (double *)malloc(sizeof(double) * numCentroids);
        k_means_y = (double *)malloc(sizeof(double) * numCentroids);
        k_means_z = (double *)malloc(sizeof(double) * numCentroids);

        for (int i = 0; i < numCentroids; i++)
        {
            k_means_x[i] = centeroids[i].x;
            k_means_y[i] = centeroids[i].y;
            k_means_z[i] = centeroids[i].z;
        }

        // Define the receiving vectors to be be big enough to hold the data. Add 1 in case there are extra data points
        recv_x = (double *)malloc(sizeof(double) * ((numElements / world_size) + 1));
        recv_y = (double *)malloc(sizeof(double) * ((numElements / world_size) + 1));
        recv_z = (double *)malloc(sizeof(double) * ((numElements / world_size) + 1));
        recv_assign = (int *)malloc(sizeof(int) * ((numElements / world_size) + 1));
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
        recv_assign = (int *)malloc(sizeof(int) * ((numElements / world_size) + 1));
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

    // For logging purposes
    // if (world_rank == 0)
    // {
    //     // Print the send counts and displacements
    //     cout << "Displacements will be : " << endl;
    //     for (int i = 0; i < world_size; ++i)
    //     {
    //         cout << displacements[i] << " ";
    //     }
    //     cout << endl;
    //     cout << "Send counts will be : " << endl;
    //     for (int i = 0; i < world_size; ++i)
    //     {
    //         cout << send_counts[i] << " ";
    //     }
    //     cout << endl;
    // }

    // TODO: Test if this can be removed
    MPI_Barrier(MPI_COMM_WORLD);

    // Scatterv for x points
    MPI_Scatterv(data_x_points, send_counts, displacements, MPI_DOUBLE,
                 recv_x, send_counts[world_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatterv for y points
    MPI_Scatterv(data_y_points, send_counts, displacements, MPI_DOUBLE,
                 recv_y, send_counts[world_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatterv for z points
    MPI_Scatterv(data_z_points, send_counts, displacements, MPI_DOUBLE,
                 recv_z, send_counts[world_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < numEpochs; ++i)
    {
        // Broadcast the centroids so everyone has updated information
        MPI_Bcast(k_means_x, numCentroids, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(k_means_y, numCentroids, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(k_means_z, numCentroids, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Every process receives a set of points and calculates the new assignments based on the new centroids
        MPI_Scatterv(k_assignment, send_counts, displacements, MPI_INT,
                     recv_assign, send_counts[world_rank], MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate the new assignments
        calculateKMean(k_means_x, k_means_y, k_means_z, recv_x, recv_y, recv_z, recv_assign, send_counts[world_rank], numCentroids);

        // Gather the point assignments back together from each process
        MPI_Gatherv(recv_assign, send_counts[world_rank], MPI_INT,
                    k_assignment, send_counts, displacements, MPI_INT, 0, MPI_COMM_WORLD);

        if (world_rank == 0)
        {
            updateCentroidDataDistributed(k_means_x, k_means_y, k_means_z, data_x_points, data_y_points, data_z_points, k_assignment, numElements, numCentroids);
        }
    }

    if (world_rank == 0)
    {
        // Log the time to finish
        double finishTime = MPI_Wtime();
        long duration = (long)((finishTime - startTime) * 1000000);
        double v = finishTime - startTime;
        // Convert the data back to a vector of points for saving
        vector<Point3D> computedPoints;
        computedPoints.reserve(numElements); // reserve space for the vector to avoid reallocation
        for (int i = 0; i < numElements; i++)
        {
            Point3D p = Point3D(data_x_points[i], data_y_points[i], data_z_points[i]);
            p.cluster = k_assignment[i];
            computedPoints.push_back(p);
        }
        saveOutputs(&computedPoints, distFilename);
        cout << "Number of Nodes: " << world_size << endl;
        printStats(numEpochs, numCentroids, &computedPoints, duration);
        areFilesEqual(serialFilename, distFilename, true);
    }
    // Clean up memory
    free(k_means_x);
    free(k_means_y);
    free(k_means_z);
    free(k_assignment);
    free(data_x_points);
    free(data_y_points);
    free(data_z_points);
    free(recv_x);
    free(recv_y);
    free(recv_z);
    free(recv_assign);
    MPI_Finalize();
}
