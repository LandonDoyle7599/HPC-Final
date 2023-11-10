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

void distributedUpdateClusters(const vector<double> &k_x, const vector<double> &k_y, const vector<double> &k_z,
                               const vector<double> &recv_x, const vector<double> &recv_y, const vector<double> &recv_z, vector<int> &assign)
{
    for (size_t i = 0; i < recv_x.size(); ++i)
    {
        double min_dist = numeric_limits<double>::max();
        int k_min_index = 0;

        for (size_t j = 0; j < k_x.size(); ++j)
        {
            double x = abs(recv_x[i] - k_x[j]);
            double y = abs(recv_y[i] - k_y[j]);
            double z = abs(recv_z[i] - k_z[j]);
            double temp_dist = (x * x) + (y * y) + (z * z);

            if (temp_dist < min_dist)
            {
                min_dist = temp_dist;
                k_min_index = j;
            }
        }
        assign[i] = k_min_index;
    }
}

void calcKmeans(vector<double> &k_means_x, vector<double> &k_means_y, vector<double> &k_means_z,
                const vector<double> &data_x_points, const vector<double> &data_y_points, const vector<double> &data_z_points, const vector<int> &k_assignment)
{
    // Iterate over the centroids and compute the means for each value
    for (size_t i = 0; i < k_means_x.size(); ++i)
    {
        double total_x = 0.0;
        double total_y = 0.0;
        double total_z = 0.0;
        int numOfpoints = 0;

        for (size_t j = 0; j < data_x_points.size(); ++j)
        {
            if (k_assignment[j] == static_cast<int>(i))
            {
                total_x += data_x_points[j];
                total_y += data_y_points[j];
                total_z += data_z_points[j];
                numOfpoints++;
            }
        }

        // Update centroid with the mean values
        if (numOfpoints != 0)
        {
            k_means_x[i] = total_x / numOfpoints;
            k_means_y[i] = total_y / numOfpoints;
            k_means_z[i] = total_z / numOfpoints;
        }
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(NULL, NULL);

    int world_size, world_rank, numEpochs, numCentroids;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    vector<double> k_means_x, k_means_y, k_means_z, data_x_points, data_y_points, data_z_points;
    vector<int> k_assignment;

    vector<double> recv_x, recv_y, recv_z;
    vector<int> recv_assign;
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

        srand(static_cast<unsigned>(time(NULL)));
        int random;
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

        recv_x.resize((data_x_points.size() / world_size) + 1);
        recv_y.resize((data_y_points.size() / world_size) + 1);
        recv_z.resize((data_z_points.size() / world_size) + 1);
        recv_assign.resize((k_assignment.size() / world_size) + 1);
    }
    else
    {
        MPI_Bcast(&numCentroids, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numEpochs, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        k_means_x.resize(numCentroids);
        k_means_y.resize(numCentroids);
        k_means_z.resize(numCentroids);

        recv_x.resize((data_x_points.size() / world_size) + 1);
        recv_y.resize((data_y_points.size() / world_size) + 1);
        recv_z.resize((data_y_points.size() / world_size) + 1);

        recv_assign.resize((k_assignment.size() / world_size) + 1);
    }

    // Scatter data across processes

    MPI_Scatter(data_x_points.data(), (data_x_points.size() / world_size) + 1, MPI_DOUBLE,
                recv_x.data(), (data_x_points.size() / world_size) + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatter(data_y_points.data(), (data_y_points.size() / world_size) + 1, MPI_DOUBLE,
                recv_y.data(), (data_y_points.size() / world_size) + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatter(data_z_points.data(), (data_z_points.size() / world_size) + 1, MPI_DOUBLE,
                recv_z.data(), (data_z_points.size() / world_size) + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int count = 0;
    auto start = chrono::high_resolution_clock::now();
    while (count < numEpochs)
    {
        MPI_Bcast(k_means_x.data(), numCentroids, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(k_means_y.data(), numCentroids, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(k_means_z.data(), numCentroids, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Scatter(k_assignment.data(), (k_assignment.size() / world_size) + 1, MPI_INT,
                    recv_assign.data(), (k_assignment.size() / world_size) + 1, MPI_INT, 0, MPI_COMM_WORLD);

        distributedUpdateClusters(k_means_x, k_means_y, k_means_z, recv_x, recv_y, recv_z, recv_assign);

        MPI_Gather(recv_assign.data(), (k_assignment.size() / world_size) + 1, MPI_INT,
                   k_assignment.data(), (k_assignment.size() / world_size) + 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (world_rank == 0)
        {
            calcKmeans(k_means_x, k_means_y, k_means_z, data_x_points, data_y_points, data_z_points, k_assignment);
        }

        count++;
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    if (world_rank == 0)
    {
        vector<Point3D> pointData;
        for (size_t i = 0; i < data_x_points.size(); ++i)
        {
            Point3D p = Point3D(data_x_points[i], data_y_points[i], data_z_points[i]);
            p.cluster = k_assignment[i];
            pointData.push_back(p)
        }

        saveOutputs(&pointData, distFilename);
        printStats(numEpochs, numCentroids, &pointData, duration.count());
        areFilesEqual(serialFilename, distFilename, true);
    }

    MPI_Finalize();

    return 0;
}
