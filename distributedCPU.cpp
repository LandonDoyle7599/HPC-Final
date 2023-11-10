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
