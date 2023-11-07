// #include "serial.hpp"
#include "serial.cpp"
// #include "gpu.cpp"

int main()
{
    int numEpochs = 100;
    int numClusters = 6;
    performSerial(numEpochs, numClusters);
    // performGPU();
}
