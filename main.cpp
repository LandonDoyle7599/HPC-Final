// #include "serial.hpp"
#include "serial.cpp"
// #include "gpu.cpp"

int main()
{
    int numEpochs = 200;
    int numClusters = 12;
    performSerial(numEpochs, numClusters);
    // performGPU();
}
