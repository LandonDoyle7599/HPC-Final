
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

const int DIMENSIONS = 3;

// Function to calculate the Euclidean distance between two points
double distance(const vector<double> &p1, const vector<double> &p2)
{
    double sum = 0;
    for (int i = 0; i < DIMENSIONS; i++)
    {
        sum += pow(p1[i] - p2[i], 2);
    }
    return sqrt(sum);
}

// Function to find the index of the nearest centroid for a data point
int nearest_centroid(const vector<double> &point, const vector<vector<double>> &centroids)
{
    int index = 0;
    double min_distance = distance(point, centroids[0]);
    for (int i = 1; i < centroids.size(); i++)
    {
        double d = distance(point, centroids[i]);
        if (d < min_distance)
        {
            index = i;
            min_distance = d;
        }
    }
    return index;
}

// Function to calculate the new centroid for a cluster
vector<double> new_centroid(const vector<vector<double>> &cluster)
{
    vector<double> centroid(DIMENSIONS, 0);
    for (int i = 0; i < cluster.size(); i++)
    {
        for (int j = 0; j < DIMENSIONS; j++)
        {
            centroid[j] += cluster[i][j];
        }
    }
    for (int j = 0; j < DIMENSIONS; j++)
    {
        centroid[j] /= cluster.size();
    }
    return centroid;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Read in the input data from a file or generate it randomly
    vector<vector<double>> data;

    if (rank == 0)
    {
        string line;
        ifstream file(filename);
        if (!file.is_open())
            cout << "Failed to open file\n";
        while (getline(file, line))
        {
            stringstream lineStream(line);
            string bit;
            double x, y, z;
            getline(lineStream, bit, ',');
            x = stof(bit);
            getline(lineStream, bit, ',');
            y = stof(bit);
            getline(lineStream, bit, '\n');
            z = stof(bit);
            data.push_back({x, y, z});
        }
    }

    // Broadcast the input data to all processes
    MPI_Bcast(&data[0][0], data.size() * DIMENSIONS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Choose k initial centroids randomly from the input data
    int k = 10;
    vector<vector<double>> centroids(k);
    if (rank == 0)
    {
        srand(time(NULL));
        for (int i = 0; i < k; i++)
        {
            int index = rand() % data.size();
            centroids[i] = data[index];
        }
    }

    // Broadcast the initial centroids to all processes
    for (int i = 0; i < k; i++)
    {
        MPI_Bcast(&centroids[i][0], DIMENSIONS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Repeat until convergence
    bool converged = false;
    while (!converged)
    {
        // Assign each data point to the nearest centroid
        vector<vector<double>> clusters(k);
        for (int i = 0; i < data.size(); i++)
        {
            int index = nearest_centroid(data[i], centroids);
            clusters[index].push_back(data[i]);
        }

        // Calculate the new centroid for each cluster
        vector<vector<double>> new_centroids(k);
        for (int i = 0; i < k; i++)
        {
            new_centroids[i] = new_centroid(clusters[i]);
        }

        // Send the new centroids to the root process
        for (int i = 0; i < k; i++)
        {
            MPI_Gather(&new_centroids[i][0], DIMENSIONS, MPI_DOUBLE, &centroids[i][0], DIMENSIONS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        // Calculate the new global centroids
        if (rank == 0)
        {
            vector<vector<double>> all_new_centroids(num_procs * k);
            MPI_Gather(&new_centroids[0][0], k * DIMENSIONS, MPI_DOUBLE, &all_new_centroids[0][0], k * DIMENSIONS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < DIMENSIONS; j++)
                {
                    double sum = 0;
                    for (int p = 0; p < num_procs; p++)
                    {
                        sum += all_new_centroids[p * k + i][j];
                    }
                    centroids[i][j] = sum / data.size();
                }
            }
        }

        // Broadcast the new global centroids to all processes
        for (int i = 0; i < k; i++)
        {
            MPI_Bcast(&centroids[i][0], DIMENSIONS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        // Check for convergence
        if (rank == 0)
        {
            converged = true;
            for (int i = 0; i < k; i++)
            {
                if (distance(new_centroids[i], centroids[i]) > 0.001)
                {
                    converged = false;
                    break;
                }
            }
        }

        // Broadcast the convergence flag to all processes
        MPI_Bcast(&converged, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    }

    // Output the final centroids and the cluster assignments for each data point
    if (rank == 0)
    {
        ofstream outfile("output.txt");
        for (int i = 0; i < k; i++)
        {
            outfile << "Centroid " << i << ": ";
            for (int j = 0; j < DIMENSIONS; j++)
            {
                outfile << centroids[i][j] << " ";
            }
            outfile << endl;
        }
        for (int i = 0; i < data.size(); i++)
        {
            int index = nearest_centroid(data[i], centroids);
            outfile << "Data point " << i << " assigned to centroid " << index << endl;
        }
        outfile.close();
    }

    MPI_Finalize();
    return 0;
}
