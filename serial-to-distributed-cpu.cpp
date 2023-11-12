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

// TODO: Remove all vectors and convert everything back to arrays. This will make it easier to send data between processes and catch compile time errors.

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

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Setup the centroids
    vector<double> k_means_x;
    vector<double> k_means_y;
    vector<double> k_means_z;
    vector<int> k_assignment; // the cluster assignment for each point

    // The data points
    vector<double> data_x_points;
    vector<double> data_y_points;
    vector<double> data_z_points;

    // The received data points
    vector<double> recv_x;
    vector<double> recv_y;
    vector<double> recv_z;
    vector<int> recv_assign;

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

        // add data from point data to the appropriate arrays
        for (size_t i = 0; i < pointData.size(); ++i)
        {
            data_x_points.push_back(pointData[i].x);
            data_y_points.push_back(pointData[i].y);
            data_z_points.push_back(pointData[i].z);
            k_assignment.push_back(0);
        }

        // Now we initialize the centroids
        vector<Point3D> centeroids = initializeCentroids(numCentroids, &pointData);

        // With pointData and centeroids we can run the serial implementation
        performSerial(numEpochs, &centeroids, &pointData, serialFilename);

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

        recv_x.resize((data_x_points.size()) + (double)world_size);
        recv_y.resize((data_y_points.size()) + (double)world_size);
        recv_z.resize((data_z_points.size()) + (double)world_size);
        recv_assign.resize((k_assignment.size()) + (double)world_size);

        // Assert the x y and z data vectors have same size
        if (data_x_points.size() != data_y_points.size() || data_x_points.size() != data_z_points.size())
        {
            cout << "Data vectors are not the same size" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Assert that the recv x y and z are at least as big as the data vectors
        if (recv_x.size() < data_x_points.size() || recv_y.size() < data_y_points.size() || recv_z.size() < data_z_points.size())
        {
            cout << "Recv vectors are not at least as big as the data vectors" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    else
    {
        // Receive the number of centroids and epochs
        MPI_Bcast(&numCentroids, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numEpochs, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Setup the k_means vectors to proper sizes
        k_means_x.resize(numCentroids);
        k_means_y.resize(numCentroids);
        k_means_z.resize(numCentroids);

        // Setup the received vectors to proper sizes, accounting for any extra data points
        recv_x.resize((data_x_points.size()) + (double)world_size);
        recv_y.resize((data_y_points.size()) + (double)world_size);
        recv_z.resize((data_z_points.size()) + (double)world_size);
        recv_assign.resize((k_assignment.size()) + (double)world_size);
    }

    // Assert recv vectors are the same size
    if (recv_x.size() != recv_y.size() || recv_x.size() != recv_z.size())
    {
        cout << "Recv vectors are not the same size" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Assert that the recv vectors are at least as big as the k assignment vector
    if (recv_assign.size < k_assignment.size())
    {
        cout << "Recv assignment vector is not at least as big as the k assignment vector" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Scatter data across processes but use displacements
    [int] send_counts(world_size);
    [int] displacements(world_size);

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
        for (int i = 0; i < displacements.count; ++i)
        {
            cout << displacements[i] << " ";
        }
        cout << endl;
        cout << "Send counts will be : " << endl;
        for (int i = 0; i < send_counts.count; ++i)
        {
            cout << send_counts[i] << " ";
        }
        cout << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    cout << "Rank : " << world_rank << " scattering x points " << endl;

    // Assert that my rank receiving x y and z are big enough for the size counts
    if (recv_x.size() < send_counts[world_rank] || recv_y.size() < send_counts[world_rank] || recv_z.size() < send_counts[world_rank])
    {
        cout << "Recv vectors are not at least as big as the send counts" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Scatterv for x points
    MPI_Scatterv(data_x_points.data(), send_counts.data(), displacements.data(), MPI_DOUBLE,
                 recv_x.data(), recv_x.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // cout << "Rank : " << world_rank << " scattering y points " << endl;

    // Scatterv for y points
    MPI_Scatterv(data_y_points.data(), send_counts.data(), displacements.data(), MPI_DOUBLE,
                 recv_y.data(), recv_y.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // cout << "Rank : " << world_rank << " scattering z points " << endl;

    // Scatterv for z points
    MPI_Scatterv(data_z_points.data(), send_counts.data(), displacements.data(), MPI_DOUBLE,
                 recv_z.data(), recv_z.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int count = 0;
    auto start = chrono::high_resolution_clock::now();
    if (world_rank == 0)
    {
        cout << "Starting k-means algorithm for " << numEpochs << " iterations...\n";
    }
    while (count < numEpochs)
    {
        // Broadcast the centroids
        MPI_Bcast(k_means_x.data(), numCentroids, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(k_means_y.data(), numCentroids, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(k_means_z.data(), numCentroids, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Scatter the assignments
        // Note: This assumes the number of centroids is evenly divisible by the number of processes
        MPI_Scatter(k_assignment.data(), (k_assignment.size() / world_size) + 1, MPI_INT,
                    recv_assign.data(), (k_assignment.size() / world_size) + 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate the new assignments
        calculateKMean(k_means_x, k_means_y, k_means_z, recv_x, recv_y, recv_z, recv_assign);

        // Gather the assignments
        MPI_Gather(recv_assign.data(), (k_assignment.size() / world_size) + 1, MPI_INT,
                   k_assignment.data(), (k_assignment.size() / world_size) + 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (world_rank == 0)
        {
            updateCentroidDataDistributed(k_means_x, k_means_y, k_means_z, data_x_points, data_y_points, data_z_points, k_assignment);
        }

        count++;
    }

    if (world_rank == 0)
    {
        // auto start = chrono::high_resolution_clock::now();
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        vector<Point3D> pointData;
        for (size_t i = 0; i < data_x_points.size(); ++i)
        {
            Point3D p = Point3D(data_x_points[i], data_y_points[i], data_z_points[i]);
            p.cluster = k_assignment[i];
            pointData.push_back(p);
        }
        saveOutputs(&pointData, distFilename);
        printStats(numEpochs, numCentroids, &pointData, duration.count());
        areFilesEqual(serialFilename, distFilename, true);
    }

    MPI_Finalize();

    return 0;
}
