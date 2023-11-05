#include "serial.cpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>

using namespace std;
// This file is the same as serial.cpp, except that it has been modified
// to use CUDA and GPU processing for the k-means.

// First we use the same readcsv function as in serial.cpp. TODO: Use the parallel version of this to read in the values
vector<Point3D> vectors = readcsv();

// Next we define a function to perform the k-means clustering on the GPU
void kMeansClusteringGPU(vector<Point3D> *points, int epochs, int k)
{
    // TODO
    return;
}
