# Spotify Visualizer

## How to Run

Note: We defaulted the setup on this repository to validate each implementation against the serial one. Because of the need for _random_ initialization in the k clusters, and the variable number of epochs and clusters, we run the serial and then execute whatever function you are using. For example, when running the single GPU implementation, we execute `main.cu` which runs the implementation serially, then the GPU implementation, so that the data can be validated.

### Serial CPU

Note: We implicity run the serial implementation for every other one as comparison. However, if you would like to run it standalone:

```bash
g++ serial-only.cpp -o serial
./serial
```

To change the number of epochs and clusters, edit the `main` function in `serial-only.cpp`.

### Parallel CPU

```bash
g++ -fopenmp serial-to-parallel.cpp -o parallel
./parallel
```

To change the number of epochs and clusters, edit the `main` function in `serial-to-parallel.cpp`.

### Distributed CPU

Running on CHPC first we need to load the module:

```bash
module load openmpi
```

When running this, you need to pass in two arguments: the number of epochs and the number of clusters. For example, to run 25 epochs with 6 clusters:

```bash
mpic++ serial-to-distributed-cpu.cpp -o distributed
mpirun -np 2 ./distributed 25 6
```

To change the number of epochs and clusters, pass in different values as arguments to the command.

`mpirun -np <number of nodes> ./distributed <number of epochs> <number of clusters>`

### Parallel GPU

Running on CHPC first we need to load the module:

```bash
module load cuda/12
```

Now we can compile and execute:

```bash
nvcc serial-to-single-gpu.cu -o gpu
./gpu
```

To change the number of epochs and clusters, edit the `main` function in `serial-to-single-gpu.cu`.

### Distributed GPU

First we setup our CHPC environment to have N nodes and each node to have a GPU. Next, we load our modules:

```bash
module load cuda/12
module load openmpi
```

Now we can compile and execute, but we need to compile the files separately and then join them together:

Path to where CUDA is installed

```bash
echo $CUDA_PATH # gets the path to where cuda is

```

Your path will be different, but it will be similar to this:

```bash
/uufs/chpc.utah.edu/sys/installdir/r8/cuda/12.2.0/
```

Then we append `lib64` to the end of that path:

```bash
/uufs/chpc.utah.edu/sys/installdir/r8/cuda/12.2.0/lib64
```

Now we can compile and execute:

```bash
nvcc -c distributedGPU.cu
mpicxx -o dist serial-to-distributed-gpu.cpp distributedGPU.o -L/uufs/chpc.utah.edu/sys/installdir/r8/cuda/12.2.0/lib64 -lcudart -lcuda
mpirun -np 2 ./dist 25 6
```

To change the number of epochs and clusters, pass in different values as arguments to the command.

`mpirun -np <number of nodes> ./distributed <number of epochs> <number of clusters>`

### Running the Python Visualization

First, edit the file `visualize.py` to point to the correct csv files you would like to visualize.

Now, run the following commands from the project root directory:

```bash
pip install -r requirements.txt
python3 visualize.py
```

## Validation

In serial.hpp we wrote a function, `areFilesEqual`, to validate two csv files against eachother. It will return true if the are, false if not. We will check every file against the ground truth, defined by the serial implementation. If we pass in the variable true after the thread names, it will print out the first 5 differences between the files and print the total number of differences.

To simplify grading and validation, we built into every implementation a function call to `performSerial` this allows the specific number of epochs and clusters to be run serially, and in the implementation. We also wrote the function `areFilesEqual` to compare the output of the serial implementation to the output of the implementation being tested.

## Our Approaches

### Serial Implementation

For the serial implementation we used the link provided by Dr. Petruzza [here](http://reasonabledeviations.com/2019/10/02/k-means-in-cpp/). We added onto this the z variable into both the Point3D value and how we update the clusters at the end of each epoch. We implemented the actual serial code in `serial.cpp` and the commonly used code across all implementations in `serial.hpp`.

### Single GPU Implementation

For the GPU implementation we took our serial implementation and changed as little as possible to preserve code modularity, but we did create a kMeansClustering kernel that takes in the data, clusters, and epochs and performs the kMeansClustering algorithm. We also create an on device function calculateDistance to easily calculate the distance between two points.

The most important thing to get right in this implementation was the memory management. We needed to allocate memory for the data, clusters, and distances. We also needed to copy the data and clusters to the device, and copy the clusters back to the host after each epoch. Keeping track of the the thread we were on within the kernel was also important, because we needed to know which cluster to update. The slowest part of this implementation was copying the data and clusters back and forth between the host and device, because we have to do that for each epoch in order to consolidate data and then fix upon completion.

### Parallel CPU Implementation

For the parallel CPU implementation we took our serial implementation and changed it to use Open MP. As with the GPU paralleization, the main consideration we had was on the nested loops iterating over every point. This was the loop we parallelized, allowing each thread to use a chunk of the total number of points (1,240,425). In the case of 5 threads, each thread works on 1240425/5 = 248085 points. We let OpenMP handle the distribution of data. After each thread is completed, we need to update the clusters, then restart the loop for the next epoch.

### Distributed CPU Implementation

For the distributed CPU implementation we wanted to be able to use as much code as possible from the serial implementation, but MPI required us to pass arrays around rather than points. This required rewriting the shared functions and creating our own distributed version called `calculateKMean` and `updateCnetroidDataDistributed`.They take in arrays of clusters, denoted with k's and arrays of the data points. Each process has its own chunk of data it is working on, called recv_x recv_y and recv_z. Each process handles their subset of the points and compares it against the clusters. After each process is done, we need to update the clusters, then restart the loop for the next epoch, and then broadcast the updated clusters to all processes.

To ensure the data could be distributed across any number of processors we used ScatterV to distribute the data and allowed each process to only work on its specified points. We had process 0 handle the serial implementation at the beginning and the comparison of points at the end to validate accuracy.

### Distributed GPU Implementation

<!-- TODO -->

## Analysis

In the data set there are 1240425 points, and the amount of data processed is equal to epochs \* numPoints. So for 100 epochs, we process 124042500 points. For 200 epochs, we process all of those points 200 times (248085000), etc.

### Unprocessed Data

![unprocessed data](./images/Unprocessed.png)

### Serial Implementation

| Time (s)  | Epochs | Clusters |
| --------- | ------ | -------- |
| 25.816402 | 100    | 6        |
| 48.143714 | 100    | 12       |
| 52.172065 | 200    | 6        |
| 95.354178 | 200    | 12       |

Serial Implementation Visualized with 6 Clusters:

![6 cluster serial CPU](./images/serialProcessed.png)

Serial Implementation Visualized with 12 Clusters:

![12 cluster serial CPU](./images/Serial-200e-12c.png)

### Single GPU Implementation

#### Device Details from Cuda Query

```text
Device 0: "Tesla T4"
  Major revision number:                         7
  Minor revision number:                         5
  Total amount of global memory:                 2770927616 bytes
  Number of multiprocessors:                     40
  Number of cores:                               320
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per block:           1024
  Maximum sizes of each dimension of a block:    1024 x 1024 x 64
  Maximum sizes of each dimension of a grid:     2147483647 x 65535 x 65535
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Clock rate:                                    1.59 GHz
  Concurrent copy and execution:                 Yes

```

Max Number of threads per block: 1024
Max Number of Blocks Per SM: 16

<!-- TODO Check whether this is strongly scalable and/or weakly scalable -->

| Time (s)  | Epochs | Clusters | Threads per Block | Blocks per Grid |
| --------- | ------ | -------- | ----------------- | --------------- |
| 6.171497  | 100    | 6        | 256               | 4704            |
| 11.557186 | 200    | 6        | 256               | 4704            |
| 5.903793  | 100    | 12       | 256               | 4704            |
| 11.755927 | 200    | 12       | 256               | 4704            |
| 35.562620 | 600    | 12       | 256               | 4704            |
| 70.860667 | 1200   | 12       | 256               | 4704            |

As you can see, compared to the serial implentation this is significantly faster. For 200 epochs on 12 clusters it took roughly 1/9th of the time.

This also shows that this algorithm is strongly scalable, because as we increase the epochs (which is a multiplier on the data we use), the time increases proportionally.

We can also change the number of threads per block to fully use the number of threads per block.

| Time(s)   | Epochs | Clusters | Threads per Block | Blocks per Grid |
| --------- | ------ | -------- | ----------------- | --------------- |
| 6.224101  | 100    | 12       | 1024              | 1176            |
| 11.819179 | 200    | 12       | 1024              | 1176            |
| 35.537325 | 600    | 12       | 1024              | 1176            |
| 70.771222 | 1200   | 12       | 1024              | 1176            |

As shown here above, fully using the threads per block did not make any noticeable speedup in the compuation. All values are within .3 seconds of their computation at 256 threads per block.

Single GPU Implementation Visualized with 6 Clusters:

![6 cluster single GPU](./images/gpuProcessed.png)

Single GPU Implementation Visualized with 12 Clusters:

![12 cluster single GPU](./images/Gpu-200e-12c.png)

### Parallel CPU Implementation

<!-- TODO Check whether this is strongly scalable and/or weakly scalable -->

This table displays scaling with an increasing number of threads while keeping the amount of data the same.

| Threads | Time (s)  | Epochs | Clusters |
| ------- | --------- | ------ | -------- |
| 4       | 27.579025 | 100    | 6        |
| 8       | 28.141354 | 100    | 6        |
| 12      | 27.901411 | 100    | 6        |
| 16      | 30.336343 | 100    | 6        |
| 24      | 29.499232 | 100    | 6        |

This table displays scaling the data keeping the number of threads the same, but increasing the amount of data (number of epochs).

| Threads | Time (s)   | Epochs | Clusters |
| ------- | ---------- | ------ | -------- |
| 12      | 27.901411  | 100    | 6        |
| 12      | 61.109881  | 200    | 6        |
| 12      | 124.803791 | 400    | 6        |
| 12      | 249.964666 | 800    | 6        |

This data tells us that this is not a strongly scalable algorithm, because as we increase the number of threads, the time does not decrease. However, as we increase the amount of data, the time does increase proportionally.

Parallel CPU Implementation Visualized with 6 Clusters:

![Parallel CPU ](./images/parallel-cpu-800e-6c.png)

### Distributed CPU Implementation

This table displays scaling with an increasing number of nodes while keeping the amount of data the same.

<!-- TODO Check whether this is strongly scalable and/or weakly scalable -->

| Nodes | Time (s) | Epochs | Clusters |
| ----- | -------- | ------ | -------- |
| 2     | 1.180753 | 25     | 4        |
| 3     | .935105  | 25     | 4        |
| 4     | .789611  | 25     | 4        |

Now with 100 epochs and 6 clusters:

| Nodes | Parallel Time (s) | Serial Time (s) | Epochs | Clusters |
| ----- | ----------------- | --------------- | ------ | -------- |
| 2     | 6.442704          | 25.327500       | 100    | 6        |
| 3     | 4.587259          | 25.238905       | 100    | 6        |
| 4     | 3.824345          | 25.277028       | 100    | 6        |

Notice how the parallel time is going down as we increase the number of nodes. This breaks up the amount of data to process per node and allows us to process the data faster.

Now with 4 nodes but scaling up the number of epochs and the amount of data:

| Nodes | Parallel Time (s) | Serial Time (s) | Epochs | Clusters |
| ----- | ----------------- | --------------- | ------ | -------- |
| 4     | 1.919520          | 12.649348       | 50     | 6        |
| 4     | 3.824345          | 25.277028       | 100    | 6        |
| 4     | 7.531646          | 50.564563       | 200    | 6        |
| 4     | 15.114408         | 10.1182546      | 400    | 6        |

A visualized example of the distributed CPU implementation with 4 nodes and 50 epochs:

![Distributed CPU](./images/Distributed-50e-6c.png)

### Distributed GPU Implementation

For our timings below we used 4 nodes, each with a GPU. That GPU on each is below:

```bash
Device 0: "NVIDIA GeForce RTX 2080 Ti"
  Major revision number:                         7
  Minor revision number:                         5
  Total amount of global memory:                 2956460032 bytes
  Number of multiprocessors:                     68
  Number of cores:                               544
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per block:           1024
  Maximum sizes of each dimension of a block:    1024 x 1024 x 64
  Maximum sizes of each dimension of a grid:     2147483647 x 65535 x 65535
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Clock rate:                                    1.54 GHz
  Concurrent copy and execution:                 Yes
```

<!-- TODO Decide whether this is strongly or weakly scaled -->

100 epochs and 6 clusters and changing number of nodes:

| Nodes | Parallel Time (s) | Serial Time (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 2     | 2.785133          | 27.798197       | 100    | 6        | 256               |
| 3     | 2.822513          | 27.794231       | 100    | 6        | 256               |
| 4     | 2.838683          | 27.768236       | 100    | 6        | 256               |

<!-- TODO Add discussion for this table -->

100 epochs and 6 clusters and same number of nodes and different threads per block:

| Nodes | Parallel Time (s) | Serial Time (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 4     | 2.785892          | 27.932044       | 100    | 6        | 64                |
| 4     | 2.921382          | 27.898281       | 100    | 6        | 256               |
| 4     | 2.844672          | 28.124140       | 100    | 6        | 1024              |

<!-- TODO Add discussion for this table -->

Now with 4 nodes but scaling up the number of epochs and the amount of data:

| Nodes | Parallel Time (s) | Serial Time (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 4     | 1.564496          | 13.977806       | 50     | 6        | 256               |
| 4     | 2.921382          | 27.898281       | 100    | 6        | 256               |
| 4     | 5.113978          | 55.535177       | 200    | 6        | 256               |
| 4     | 9.965394          | 111.073740      | 400    | 6        | 256               |

<!-- TODO: Add comparison between this and the CPU and Single GPU Implementations. (THis one is faster by a few seconds) -->

## References

- [Initial Setup](http://reasonabledeviations.com/2019/10/02/k-means-in-cpp/)
- [K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
- [Paper on K-Means](https://arxiv.org/pdf/2203.01081.pdf)
- [Distributed GPUs](https://docs.ccv.brown.edu/oscar/gpu-computing/mpi-cuda)
