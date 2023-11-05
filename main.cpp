#include "serial.hpp"
// #include "gpu.cpp"





int main()
{
    int numberEpochs = 100;
    int numberClusters = 6;
    // performSerial(numberEpochs, numberClusters);
    // performGPU();
    bool res = areFilesEqual("single-gpu.csv", "serialOutput.csv", true);
    std::cout << "Testing: " <<  res << std::endl;
}
