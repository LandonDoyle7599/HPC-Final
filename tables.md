## Analysis of Parallel Implementations

### Serial Implementation

| Time (s)  | Epochs | Clusters |
| --------- | ------ | -------- |
| 9.807173  | 50     | 3        |
| 12.552126 | 50     | 4        |
| 17.976485 | 50     | 6        |
| 19.570529 | 100    | 3        |
| 25.194927 | 100    | 4        |
| 35.975967 | 100    | 6        |
| 38.853329 | 200    | 3        |
| 50.216369 | 200    | 4        |
| 72.608509 | 200    | 6        |

### Parallel CPU Implementation

TODO Aaron: Timing are done, just need speedup and efficiency calculations.

Increase processes, keeping epochs the same is strong scaling.

| Threads | Time Parallel (s) | Time Serial (s) | Epochs | Clusters | Speedup | Efficiency |
| ------- | ----------------- | --------------- | ------ | -------- | ------- | ---------- |
| 4       | 10.889288         | 19.551173       | 100    | 3        |         |            |
| 8       | 8.705353          | 19.522629       | 100    | 3        |         |            |
| 16      | 88.33985          | 19.614534       | 100    | 3        |         |            |
| 4       | 12.073964         | 25.194927       | 100    | 4        |         |            |
| 8       | 9.659430          | 25.156176       | 100    | 4        |         |            |
| 16      | 10.108751         | 25.094426       | 100    | 4        |         |            |
| 4       | 20.276805         | 36.017955       | 100    | 6        |         |            |
| 8       | 12.101930         | 35.999700       | 100    | 6        |         |            |
| 16      | 12.805487         | 36.019349       | 100    | 6        |         |            |

<!-- TODO Add discussion of results -->

Increase epochs and threads proportionally is weak scaling.

| Threads | Time (s)  | Time Serial (s) | Epochs | Clusters | Speedup | Efficiency |
| ------- | --------- | --------------- | ------ | -------- | ------- | ---------- |
| 4       | 6.825731  | 9.807173        | 50     | 3        |         |            |
| 8       | 8.923136  | 19.570529       | 100    | 3        |         |            |
| 16      | 17.886764 | 38.853329       | 200    | 3        |         |            |
| 4       | 7.926284  | 12.552126       | 50     | 4        |         |            |
| 8       | 9.594114  | 25.115965       | 100    | 4        |         |            |
| 16      | 19.675230 | 50.216369       | 200    | 4        |         |            |
| 4       | 10.002409 | 17.976485       | 50     | 6        |         |            |
| 8       | 11.812873 | 35.975967       | 100    | 6        |         |            |
| 16      | 25.851216 | 72.608509       | 200    | 6        |         |            |

<!-- TODO Add discussion of results -->

### Single GPU Implementation

#### Analytical Discussion

We are looking at the following GPU:

```text
Device 0: "NVIDIA GeForce RTX 3090"
  Major revision number:                         8
  Minor revision number:                         6
  Total amount of global memory:                 3963289600 bytes
  Number of multiprocessors:                     82
  Number of cores:                               656
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per block:           1024
  Maximum sizes of each dimension of a block:    1024 x 1024 x 64
  Maximum sizes of each dimension of a grid:     2147483647 x 65535 x 65535
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Clock rate:                                    1.70 GHz
  Concurrent copy and execution:                 Yes
```

Becuase it can have 1024 threads per block, we will use that as our baseline. It has 82 multiprocesors, and ideally we hit our maximum number of threads and use all of the multiprocessors.

#### Experimental Results

Increase processes, keeping epochs the same is strong scaling.

<!-- TODO: Aaron add speedup and efficiency -->

| Threads per Block | Time (s) | Time Serial (s) | Epochs | Clusters | Speedup | Efficiency |
| ----------------- | -------- | --------------- | ------ | -------- | ------- | ---------- |
| 256               | 1.726700 | 9.633755        | 50     | 3        |         |            |
| 512               | 1.678286 | 9.381626        | 50     | 3        |         |            |
| 1024              | 1.838031 | 9.616937        | 50     | 3        |         |            |
| 256               | 1.731712 | 12.514634       | 50     | 4        |         |            |
| 512               | 1.718942 | 12.332335       | 50     | 4        |         |            |
| 1024              | 1652933  | 12.598727       | 50     | 4        |         |            |
| 256               | 1.838919 | 18.889932       | 50     | 6        |         |            |
| 512               | 1.670835 | 19.385700       | 50     | 6        |         |            |
| 1024              | 1.717685 | 18.611738       | 50     | 6        |         |            |

<!-- TODO: Add discussion for this table -->

Increase epochs and threads proportionally is weak scaling.

<!-- TODO: Aaron add speedup and efficiency -->

| Threads per Block | Time (s) | Time Serial (s) | Epochs | Clusters | Speedup | Efficiency |
| ----------------- | -------- | --------------- | ------ | -------- | ------- | ---------- |
| 256               | 1.726700 | 9.633755        | 50     | 3        |         |            |
| 512               | 3.224774 | 19.231200       | 100    | 3        |         |            |
| 1024              | 6.292316 | 38.413889       | 200    | 3        |         |            |
| 256               | 1.731712 | 12.514634       | 50     | 4        |         |            |
| 512               | 3.190473 | 24.911140       | 100    | 4        |         |            |
| 1024              | 6.146267 | 50.652925       | 200    | 4        |         |            |
| 256               | 1.838919 | 18.889932       | 50     | 6        |         |            |
| 512               | 3.311074 | 37.237659       | 100    | 6        |         |            |
| 1024              | 6.135140 | 75.330369       | 200    | 6        |         |            |

<!-- TODO: Add discussion for this table -->

## Analysis of Distributed Implementations

### Distributed CPU Implementation

On 3 clusters:

| Nodes | Time (s) | Time Serial (s) | Epochs | Clusters |
| ----- | -------- | --------------- | ------ | -------- |
| 2     | 0.766036 | 4.811508        | 25     | 3        |
| 3     | 0.621373 | 4.827385        | 25     | 3        |
| 4     | 0.581396 | 4.842869        | 25     | 3        |

On 4 clusters:

| Nodes | Time (s) | Time Serial (s) | Epochs | Clusters |
| ----- | -------- | --------------- | ------ | -------- |
| 2     | 0.947897 | 6.203727        | 25     | 4        |
| 3     | 0.739652 | 6.202104        | 25     | 4        |
| 4     | 0.670900 | 6.228862        | 25     | 4        |

On 6 Clusters:

| Nodes | Parallel Time (s) | Time Serial (s) | Epochs | Clusters |
| ----- | ----------------- | --------------- | ------ | -------- |
| 2     | 1.245606          | 8.818218        | 25     | 6        |
| 3     | 0.942043          | 8.863819        | 25     | 6        |
| 4     | 0.819143          | 8.859402        | 25     | 6        |

Notice how the parallel time is going down as we increase the number of nodes. This breaks up the amount of data to process per node and allows us to process the data faster.

Now with 4 nodes but scaling up the number of epochs and the amount of data:

| Nodes | Parallel Time (s) | Time Serial (s) | Epochs | Clusters |
| ----- | ----------------- | --------------- | ------ | -------- |
| 4     | 0.820076          | 8.816901        | 25     | 6        |
| 4     | 1.595893          | 17.698479       | 50     | 6        |
| 4     | 3.121529          | 35.421519       | 100    | 6        |
| 4     | 6.180246          | 70.824262       | 200    | 6        |

### Distributed GPU Implementation

On 3 Clusters:

| Nodes | Parallel Time (s) | Time Serial (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 2     | 0.707328          | 4.711447        | 25     | 3        | 256               |
| 3     | 0.647141          | 4.714513        | 25     | 3        | 256               |
| 4     | 0.799928          | 4.704620        | 25     | 3        | 256               |

On 4 Clusters:

| Nodes | Parallel Time (s) | Time Serial (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 2     | 0.634951          | 6.065612        | 25     | 4        | 256               |
| 3     | 0.814948          | 6.070404        | 25     | 4        | 256               |
| 4     | 0.804700          | 6.096105        | 25     | 4        | 256               |

On 6 Clusters:

| Nodes | Parallel Time (s) | Time Serial (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 2     | 0.778927          | 8.686541        | 25     | 6        | 256               |
| 3     | 0.657219          | 8.690562        | 25     | 6        | 256               |
| 4     | 0.821848          | 8.679386        | 25     | 6        | 256               |

Here we see that as we increase the number of nodes the time does go down for 3, but jumps back up to 4. This could be due to the amount of overhead involved in running this on only 25 epochs. As we increase the number of epochs from 25 to 100 we see much clearer the advantage of using more nodes.

To illustrate the point, here are the results for 100 epochs and 6 clusters:

| Nodes | Parallel Time (s) | Time Serial (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 2     | 1.847004          | 34.788968       | 100    | 6        | 256               |
| 3     | 1.797190          | 34.751091       | 100    | 6        | 256               |
| 4     | 2.352228          | 34.801689       | 100    | 6        | 256               |

<!-- TODO: Landon Write discussion about how the distributed CPU and distributed GPU compare -->

Now with 4 nodes but scaling up the number of epochs

| Nodes | Parallel Time (s) | Time Serial (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 4     | 0.821734          | 8.690958        | 25     | 6        | 256               |
| 4     | 1.296684          | 17.392156       | 50     | 6        | 256               |
| 4     | 2.257343          | 34.808612       | 100    | 6        | 256               |
| 4     | 4.277128          | 69.621106       | 200    | 6        | 256               |

On this table, we see that the increasing epochs does increase the time, and doubling the amount of data very nearly doubles the time. This is to be expected.

100 epochs and 6 clusters and same number of nodes and different threads per block:

| Nodes | Parallel Time (s) | Time Serial (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 4     | 1.400168          | 17.441074       | 50     | 6        | 64                |
| 4     | 1.296684          | 17.392156       | 50     | 6        | 256               |
| 4     | 1.410010          | 17.333747       | 50     | 6        | 1024              |

Note: For these 4 nodes, we have the following GPU configs:

<!-- TODO Landon: Write analytical discussion for the GPUs and ideal block size  -->

```bash
Device 0: "NVIDIA GeForce RTX 3090"
  Major revision number:                         8
  Minor revision number:                         6
  Total amount of global memory:                 3963289600 bytes
  Number of multiprocessors:                     82
  Number of cores:                               656
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per block:           1024
  Maximum sizes of each dimension of a block:    1024 x 1024 x 64
  Maximum sizes of each dimension of a grid:     2147483647 x 65535 x 65535
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Clock rate:                                    1.70 GHz
  Concurrent copy and execution:                 Yes

Device 1: "NVIDIA GeForce RTX 3090"
  Major revision number:                         8
  Minor revision number:                         6
  Total amount of global memory:                 3963289600 bytes
  Number of multiprocessors:                     82
  Number of cores:                               656
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per block:           1024
  Maximum sizes of each dimension of a block:    1024 x 1024 x 64
  Maximum sizes of each dimension of a grid:     2147483647 x 65535 x 65535
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Clock rate:                                    1.70 GHz
  Concurrent copy and execution:                 Yes

Device 2: "NVIDIA GeForce RTX 3090"
  Major revision number:                         8
  Minor revision number:                         6
  Total amount of global memory:                 3963289600 bytes
  Number of multiprocessors:                     82
  Number of cores:                               656
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per block:           1024
  Maximum sizes of each dimension of a block:    1024 x 1024 x 64
  Maximum sizes of each dimension of a grid:     2147483647 x 65535 x 65535
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Clock rate:                                    1.70 GHz
  Concurrent copy and execution:                 Yes

Device 3: "NVIDIA GeForce RTX 3090"
  Major revision number:                         8
  Minor revision number:                         6
  Total amount of global memory:                 3963289600 bytes
  Number of multiprocessors:                     82
  Number of cores:                               656
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per block:           1024
  Maximum sizes of each dimension of a block:    1024 x 1024 x 64
  Maximum sizes of each dimension of a grid:     2147483647 x 65535 x 65535
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Clock rate:                                    1.70 GHz
  Concurrent copy and execution:                 Yes
```
