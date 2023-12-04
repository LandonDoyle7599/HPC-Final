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

TODO: Timing are done, just need speedup and efficiency calculations.

Increase processes, keeping epochs the same is strong scaling.

| Threads | Time Parallel (s) | Serial Time (s) | Epochs | Clusters | Speedup | Efficiency |
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

Increase epochs and threads proportionally is weak scaling.

| Threads | Time (s)  | Serial Time | Epochs | Clusters | Speedup | Efficiency |
| ------- | --------- | ----------- | ------ | -------- | ------- | ---------- |
| 4       | 6.825731  | 9.807173    | 50     | 3        |         |            |
| 8       | 8.923136  | 19.570529   | 100    | 3        |         |            |
| 16      | 17.886764 | 38.853329   | 200    | 3        |         |            |
| 4       | 7.926284  | 12.552126   | 50     | 4        |         |            |
| 8       | 9.594114  | 25.115965   | 100    | 4        |         |            |
| 16      | 19.675230 | 50.216369   | 200    | 4        |         |            |
| 4       | 10.002409 | 17.976485   | 50     | 6        |         |            |
| 8       | 11.812873 | 35.975967   | 100    | 6        |         |            |
| 16      | 25.851216 | 72.608509   | 200    | 6        |         |            |

### Single GPU Implementation

TODO

Increase processes, keeping epochs the same is strong scaling.

| Threads per Block | Blocks per Grid | Time (s)  | Epochs | Clusters | Speedup | Efficiency |
| ----------------- | --------------- | --------- | ------ | -------- | ------- | ---------- |
| 256               | xxxx            | xxxxxxxx  | 100    | 3        |         |            |
| 512               | xxxx            | xxxxxxxxx | 100    | 3        |         |            |
| 1024              | xxxx            | xxxxxxxx  | 100    | 3        |         |            |
| 256               | xxxx            | xxxxxxxx  | 100    | 6        |         |            |
| 512               | xxxx            | xxxxxxxxx | 100    | 6        |         |            |
| 1024              | xxxx            | xxxxxxxx  | 100    | 6        |         |            |
| 256               | xxxx            | xxxxxxxxx | 100    | 12       |         |            |
| 512               | xxxx            | xxxxxxxxx | 100    | 12       |         |            |
| 1024              | xxxx            | xxxxxxxxx | 100    | 12       |         |            |

Increase epochs and threads proportionally is weak scaling.

| Threads per Block | Blocks per Grid | Time (s)  | Epochs | Clusters | Speedup | Efficiency |
| ----------------- | --------------- | --------- | ------ | -------- | ------- | ---------- |
| 256               | xxxx            | xxxxxxxx  | 100    | 3        |         |            |
| 512               | xxxx            | xxxxxxxxx | 200    | 3        |         |            |
| 1024              | xxxx            | xxxxxxxx  | 400    | 3        |         |            |
| 256               | xxxx            | xxxxxxxx  | 100    | 6        |         |            |
| 512               | xxxx            | xxxxxxxxx | 200    | 6        |         |            |
| 1024              | xxxx            | xxxxxxxx  | 400    | 6        |         |            |
| 256               | xxxx            | xxxxxxxxx | 100    | 12       |         |            |
| 512               | xxxx            | xxxxxxxxx | 200    | 12       |         |            |
| 1024              | xxxx            | xxxxxxxxx | 400    | 12       |         |            |

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

| Nodes | Parallel Time (s) | Serial Time (s) | Epochs | Clusters |
| ----- | ----------------- | --------------- | ------ | -------- |
| 2     | 1.245606          | 8.818218        | 25     | 6        |
| 3     | 0.942043          | 8.863819        | 25     | 6        |
| 4     | 0.819143          | 8.859402        | 25     | 6        |

Notice how the parallel time is going down as we increase the number of nodes. This breaks up the amount of data to process per node and allows us to process the data faster.

Now with 4 nodes but scaling up the number of epochs and the amount of data:

| Nodes | Parallel Time (s) | Serial Time (s) | Epochs | Clusters |
| ----- | ----------------- | --------------- | ------ | -------- |
| 4     | 0.820076          | 8.816901        | 25     | 6        |
| 4     | 1.595893          | 17.698479       | 50     | 6        |
| 4     | 3.121529          | 35.421519       | 100    | 6        |
| 4     | 6.180246          | 70.824262       | 200    | 6        |

### Distributed GPU Implementation

On 3 Clusters:

| Nodes | Parallel Time (s) | Serial Time (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 2     | 0.707328          | 4.711447        | 25     | 3        | 256               |
| 3     | 0.647141          | 4.714513        | 25     | 3        | 256               |
| 4     | 0.799928          | 4.704620        | 25     | 3        | 256               |

On 4 Clusters:

| Nodes | Parallel Time (s) | Serial Time (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 2     | 0.634951          | 6.065612        | 25     | 4        | 256               |
| 3     | 0.814948          | 6.070404        | 25     | 4        | 256               |
| 4     | 0.804700          | 6.096105        | 25     | 4        | 256               |

On 6 Clusters:

| Nodes | Parallel Time (s) | Serial Time (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 2     | 0.778927          | 8.686541        | 25     | 6        | 256               |
| 3     | 0.657219          | 8.690562        | 25     | 6        | 256               |
| 4     | 0.821848          | 8.679386        | 25     | 6        | 256               |

<!-- TODO Add discussion for these tables above ^ -->
<!-- TODO: Landon Write discussion about how the distributed CPU and distributed GPU compare -->

Now with 4 nodes but scaling up the number of epochs

| Nodes | Parallel Time (s) | Serial Time (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 4     | 0.821734          | 8.690958        | 25     | 6        | 256               |
| 4     | 1.296684          | 17.392156       | 50     | 6        | 256               |
| 4     | 2.257343          | 34.808612       | 100    | 6        | 256               |
| 4     | 4.277128          | 69.621106       | 200    | 6        | 256               |

<!-- TODO Add discussion for this table above^ -->

100 epochs and 6 clusters and same number of nodes and different threads per block:

| Nodes | Parallel Time (s) | Serial Time (s) | Epochs | Clusters | Threads per Block |
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
