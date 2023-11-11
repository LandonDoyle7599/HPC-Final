#include <iostream>
#include <vector>
#include <mpi.h>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;

struct Point3D
{
    double x, y, z;

    Point3D() : x(0), y(0), z(0) {}

    Point3D(double x, double y, double z) : x(x), y(y), z(z) {}

    double distance(const Point3D &other) const
    {
        return sqrt(pow(x - other.x, 2) + pow(y - other.y, 2) + pow(z - other.z, 2));
    }
};

vector<Point3D> readcsv(const string filename)
{
    vector<Point3D> points;
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
        points.push_back(Point3D(x, y, z));
    }
    return points;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int k = 3; // Number of clusters

    if (rank == 0)
    {
        // Master process reads data and distributes it to other processes
        string filename = "song_data.csv";
        vector<Point3D> all_points = readcsv(filename);

        if (all_points.size() < k * size)
        {
            cerr << "Error: Insufficient data points for the specified number of processes and clusters.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Scatter data to all processes
        vector<Point3D> local_points(all_points.size() / size);
        MPI_Scatter(all_points.data(), local_points.size() * sizeof(Point3D), MPI_BYTE,
                    local_points.data(), local_points.size() * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Perform K-means clustering
        vector<Point3D> centroids(k);
        // Initialize centroids (you can choose a better initialization method)
        for (int i = 0; i < k; ++i)
        {
            centroids[i] = local_points[i];
        }

        bool converged = false;
        while (!converged)
        {
            // Assign each point to the nearest centroid
            vector<vector<Point3D>> clusters(k);
            for (const auto &point : local_points)
            {
                int closest_centroid = 0;
                double min_distance = numeric_limits<double>::max();
                for (int i = 0; i < k; ++i)
                {
                    double distance = point.distance(centroids[i]);
                    if (distance < min_distance)
                    {
                        min_distance = distance;
                        closest_centroid = i;
                    }
                }
                clusters[closest_centroid].push_back(point);
            }

            // Calculate new centroids
            vector<Point3D> new_centroids(k);
            for (int i = 0; i < k; ++i)
            {
                if (!clusters[i].empty())
                {
                    double sum_x = 0, sum_y = 0, sum_z = 0;
                    for (const auto &point : clusters[i])
                    {
                        sum_x += point.x;
                        sum_y += point.y;
                        sum_z += point.z;
                    }
                    new_centroids[i] = Point3D(sum_x / clusters[i].size(), sum_y / clusters[i].size(), sum_z / clusters[i].size());
                }
                else
                {
                    // If a cluster is empty, keep the centroid the same
                    new_centroids[i] = centroids[i];
                }
            }

            // Check for convergence
            converged = true;
            for (int i = 0; i < k; ++i)
            {
                if (new_centroids[i].distance(centroids[i]) > 1e-6)
                {
                    converged = false;
                    break;
                }
            }

            // Update centroids for the next iteration
            centroids = new_centroids;

            // Broadcast convergence status to all processes
            MPI_Bcast(&converged, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

            // Gather updated centroids from all processes
            MPI_Gather(centroids.data(), k * sizeof(Point3D), MPI_BYTE,
                       centroids.data(), k * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);
        }

        // Output final centroids
        if (rank == 0)
        {
            cout << "Final centroids:\n";
            for (int i = 0; i < k; ++i)
            {
                cout << "Centroid " << i + 1 << ": (" << centroids[i].x << ", " << centroids[i].y << ", " << centroids[i].z << ")\n";
            }
        }
    }
    else
    {
        // Worker processes receive local data and perform clustering until convergence
        vector<Point3D> local_points(all_points.size() / size);
        MPI_Scatter(nullptr, local_points.size() * sizeof(Point3D), MPI_BYTE,
                    local_points.data(), local_points.size() * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);

        vector<Point3D> centroids(k);
        // Initialize centroids (you can choose a better initialization method)
        for (int i = 0; i < k; ++i)
        {
            centroids[i] = local_points[i];
        }

        bool converged = false;
        while (!converged)
        {
            // Assign each point to the nearest centroid
            vector<vector<Point3D>> clusters(k);
            for (const auto &point : local_points)
            {
                int closest_centroid = 0;
                double min_distance = numeric_limits<double>::max();
                for (int i = 0; i < k; ++i)
                {
                    double distance = point.distance(centroids[i]);
                    if (distance < min_distance)
                    {
                        min_distance = distance;
                        closest_centroid = i;
                    }
                }
                clusters[closest_centroid].push_back(point);
            }

            // Calculate new centroids
            vector<Point3D> new_centroids(k);
            for (int i = 0; i < k; ++i)
            {
                if (!clusters[i].empty())
                {
                    double sum_x = 0, sum_y = 0, sum_z = 0;
                    for (const auto &point : clusters[i])
                    {
                        sum_x += point.x;
                        sum_y += point.y;
                        sum_z += point.z;
                    }
                    new_centroids[i] = Point3D(sum_x / clusters[i].size(), sum_y / clusters[i].size(), sum_z / clusters[i].size());
                }
                else
                {
                    // If a cluster is empty, keep the centroid the same
                    new_centroids[i] = centroids[i];
                }
            }

            // Check for convergence
            converged = true;
            for (int i = 0; i < k; ++i)
            {
                if (new_centroids[i].distance(centroids[i]) > 1e-6)
                {
                    converged = false;
                    break;
                }
            }

            // Update centroids for the next iteration
            centroids = new_centroids;

            // Broadcast convergence status to all processes
            MPI_Bcast(&converged, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

            // Gather updated centroids from all processes
            MPI_Gather(centroids.data(), k * sizeof(Point3D), MPI_BYTE,
                       centroids.data(), k * sizeof(Point3D), MPI_BYTE, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
