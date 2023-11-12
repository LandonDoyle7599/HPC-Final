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

void calculateKMean(const vector<double> &k_x, const vector<double> &k_y, const vector<double> &k_z,
                    const vector<double> &recv_x, const vector<double> &recv_y, const vector<double> &recv_z, vector<int> &assign)
{
    for (int i = 0; i < recv_x.size(); ++i)
    {
        double min_dist = numeric_limits<double>::max();
        int clusterID = 0;
        for (int j = 0; j < k_x.size(); ++j)
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
        assign[i] = clusterID;
    }
}

void updateCentroidDataDistributed(vector<double> &k_means_x, vector<double> &k_means_y, vector<double> &k_means_z,
                                   const vector<double> &data_x_points, const vector<double> &data_y_points, const vector<double> &data_z_points, const vector<int> &k_assignment)
{
    int numK = k_means_x.size();
    vector<int> nPoints(numK, 0);
    vector<double> sumX(numK, 0.0);
    vector<double> sumY(numK, 0.0);

    // Iterate over the centroids and compute the means for each value
    for (int i = 0; i < data_x_points.size(); ++i)
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
    int world_size;
    int world_rank;
    int numElements;
    double startTime;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Setup the centroids (the final output)
    vector<double> k_means_x;
    vector<double> k_means_y;
    vector<double> k_means_z;
    vector<int> k_assignment; // the cluster assignment for each point put together

    // The data points
    vector<double> data_x_points;
    vector<double> data_y_points;
    vector<double> data_z_points;

    // The received data points (local data)
    vector<double> recv_x;
    vector<double> recv_y;
    vector<double> recv_z;
    vector<int> recv_assign; // the cluster assignment for each point for the local data

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
        numElements = pointData.size();
        MPI_Bcast(&numElements, 1, MPI_INT, 0, MPI_COMM_WORLD);
        k_assignment.resize(numElements);
        // add data from point data to the appropriate arrays
        for (size_t i = 0; i < pointData.size(); ++i)
        {
            data_x_points.push_back(pointData[i].x);
            data_y_points.push_back(pointData[i].y);
            data_z_points.push_back(pointData[i].z);
            k_assignment[i] = 0;
        }

        // Now we initialize the centroids
        vector<Point3D> centeroids = initializeCentroids(numCentroids, &pointData);

        // With pointData and centeroids we can run the serial implementation
        performSerial(numEpochs, &centeroids, &pointData, serialFilename);

        startTime = MPI_Wtime();

        // Setup the k_means vectors to proper sizes
        k_means_x.resize(numCentroids);
        k_means_y.resize(numCentroids);
        k_means_z.resize(numCentroids);

        for (int i = 0; i < numCentroids; ++i)
        {
            k_means_x[i] = centeroids[i].x;
            k_means_y[i] = centeroids[i].y;
            k_means_z[i] = centeroids[i].z;
        }

        cout << "Running k-means algorithm for " << numEpochs << " iterations...\n";

        // Define the receiving vectors to be be big enough to hold the data. Add 1 in case there are extra data points
        recv_x.resize((numElements / world_size) + 1);
        recv_y.resize((numElements / world_size) + 1);
        recv_z.resize((numElements / world_size) + 1);
        recv_assign.resize((numElements / world_size) + 1);

        // Assert the x y and z data vectors have same size
        if (data_x_points.size() != data_y_points.size() || data_x_points.size() != data_z_points.size())
        {
            cout << "Data vectors are not the same size" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    else
    {
        // Receive the number of centroids and epochs
        MPI_Bcast(&numCentroids, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numEpochs, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Setup the k_means vectors to proper sizes
        k_means_x.resize(numCentroids);
        k_means_y.resize(numCentroids);
        k_means_z.resize(numCentroids);

        // Setup the received vectors to proper sizes, accounting for any extra data points
        recv_x.resize((numElements / world_size) + 1);
        recv_y.resize((numElements / world_size) + 1);
        recv_z.resize((numElements / world_size) + 1);
        recv_assign.resize((numElements / world_size) + 1);
    }

    // Assert recv vectors are at least as big as numElements
    if (recv_x.size() < numElements / world_size || recv_y.size() < numElements / world_size || recv_z.size() < numElements / world_size || recv_assign.size() < numElements / world_size)
    {
        cout << "Recv vectors are not at least as big as their proportion" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Scatter data across processes but use displacements
    int send_counts[world_size];
    int displacements[world_size];

    // Break up the data into chunks accouting for evenly dividing the data
    for (int i = 0; i < world_size; ++i)
    {
        send_counts[i] = data_x_points.size() / world_size;
        displacements[i] = i * send_counts[i];
    }
    // Add the remainder to each process and adjust the displacements
    for (int i = 0; i < data_x_points.size() % world_size; ++i)
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

    // cout << "Rank : " << world_rank << " scattering points " << endl;

    // Assert that my rank receiving x y and z are big enough for the size counts
    if (recv_x.size() < send_counts[world_rank] || recv_y.size() < send_counts[world_rank] || recv_z.size() < send_counts[world_rank])
    {
        cout << "Recv vectors are not at least as big as the send counts" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Scatterv for x points
    MPI_Scatterv(data_x_points.data(), send_counts, displacements, MPI_DOUBLE,
                 recv_x.data(), recv_x.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatterv for y points
    MPI_Scatterv(data_y_points.data(), send_counts, displacements, MPI_DOUBLE,
                 recv_y.data(), recv_y.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatterv for z points
    MPI_Scatterv(data_z_points.data(), send_counts, displacements, MPI_DOUBLE,
                 recv_z.data(), recv_z.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Start the timer using MPI https://www.mcs.anl.gov/research/projects/mpi/tutorial/gropp/node139.html#:~:text=The%20elapsed%20(wall%2Dclock),n%22%2C%20t2%20%2D%20t1%20)%3B

    if (world_rank == 0)
    {
        cout << "Starting k-means algorithm for " << numEpochs << " iterations...\n";
    }
    for (int i = 0; i < numEpochs; ++i)
    {
        // Broadcast the centroids
        MPI_Bcast(k_means_x.data(), numCentroids, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(k_means_y.data(), numCentroids, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(k_means_z.data(), numCentroids, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Scatter the assignments
        // Note: This assumes the number of centroids is evenly divisible by the number of processes
        MPI_Scatter(k_assignment.data(), (numElements / world_size) + 1, MPI_INT,
                    recv_assign.data(), (numElements / world_size) + 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate the new assignments
        calculateKMean(k_means_x, k_means_y, k_means_z, recv_x, recv_y, recv_z, recv_assign);

        // Gather the assignments
        MPI_Gather(recv_assign.data(), (numElements / world_size) + 1, MPI_INT,
                   k_assignment.data(), (numElements / world_size) + 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (world_rank == 0)
        {
            updateCentroidDataDistributed(k_means_x, k_means_y, k_means_z, data_x_points, data_y_points, data_z_points, k_assignment);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        double finishTime = MPI_Wtime();
        long duration = (long)((finishTime - startTime) * 1000);
        double v = finishTime - startTime;
        cout << "Time: " << v << " ms" << endl;
        vector<Point3D> pointData[numElements];

        // Validate xpoints, ypints, zpoints and k_assignment are the same size
        if (data_x_points.size() != data_y_points.size() || data_x_points.size() != data_z_points.size() || data_x_points.size() != k_assignment.size())
        {
            cout << "Data vectors are not the same size" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < numElements; i++)
        {
            Point3D p = Point3D(data_x_points[i], data_y_points[i], data_z_points[i]);
            p.cluster = k_assignment[i];
            pointData[i] = p;
        }
        saveOutputs(&pointData, distFilename);
        printStats(numEpochs, numCentroids, &pointData, duration);
        areFilesEqual(serialFilename, distFilename, true);
    }
    // Clean up by deallocating memory
    k_means_x.clear();
    k_means_y.clear();
    k_means_z.clear();
    k_assignment.clear();
    data_x_points.clear();
    data_y_points.clear();
    data_z_points.clear();
    recv_x.clear();
    recv_y.clear();
    recv_z.clear();
    recv_assign.clear();

    MPI_Finalize();
}
