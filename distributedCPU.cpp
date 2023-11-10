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

// Define MPI datatype for Point3D
MPI_Datatype createPoint3DType()
{
    const int nitems = 4;                                                  // Number of items in the structure
    int blocklengths[4] = {1, 1, 1, 1};                                    // Number of elements for each item
    MPI_Datatype types[4] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT}; // Datatypes of each item

    MPI_Aint offsets[4]; // Offsets of each item in the structure
    offsets[0] = offsetof(Point3D, x);
    offsets[1] = offsetof(Point3D, y);
    offsets[2] = offsetof(Point3D, z);
    offsets[3] = offsetof(Point3D, cluster);

    MPI_Datatype mpi_point_type;
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_point_type);
    MPI_Type_commit(&mpi_point_type);

    return mpi_point_type;
}