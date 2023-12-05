## Analysis of Parallel Implementations

### Serial Implementation

Table 1

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

Table 2

| Threads | Time Parallel (s) | Time Serial (s) | Epochs | Clusters | Speedup | Efficiency |
| ------- | ----------------- | --------------- | ------ | -------- | ------- | ---------- |
| 4       | 10.889288         | 19.551173       | 100    | 3        | 1.79545 | 0.44886    |
| 8       | 8.705353          | 19.522629       | 100    | 3        | 2.24260 | 0.28033    |
| 16      | 8.833985          | 19.614534       | 100    | 3        | 2.22035 | 0.13877    |
| 4       | 12.073964         | 25.194927       | 100    | 4        | 2.08671 | 0.52168    |
| 8       | 9.659430          | 25.156176       | 100    | 4        | 2.60431 | 0.32554    |
| 16      | 10.108751         | 25.094426       | 100    | 4        | 2.48245 | 0.15515    |
| 4       | 20.276805         | 36.017955       | 100    | 6        | 1.77631 | 0.44408    |
| 8       | 12.101930         | 35.999700       | 100    | 6        | 2.97471 | 0.37184    |
| 16      | 12.805487         | 36.019349       | 100    | 6        | 2.81281 | 0.17580    |

Increase epochs and threads proportionally is weak scaling.

Table 3

| Threads | Time (s)  | Time Serial (s) | Epochs | Clusters | Speedup | Efficiency |
| ------- | --------- | --------------- | ------ | -------- | ------- | ---------- |
| 4       | 6.825731  | 9.807173        | 50     | 3        | 1.43679 | 0.35920    |
| 8       | 8.923136  | 19.570529       | 100    | 3        | 2.19323 | 0.27415    |
| 16      | 17.886764 | 38.853329       | 200    | 3        | 2.17218 | 0.13576    |
| 4       | 7.926284  | 12.552126       | 50     | 4        | 1.58361 | 0.39590    |
| 8       | 9.594114  | 25.115965       | 100    | 4        | 2.61785 | 0.32723    |
| 16      | 19.675230 | 50.216369       | 200    | 4        | 2.55226 | 0.15952    |
| 4       | 10.002409 | 17.976485       | 50     | 6        | 1.79722 | 0.44930    |
| 8       | 11.812873 | 35.975967       | 100    | 6        | 3.04549 | 0.38069    |
| 16      | 25.851216 | 72.608509       | 200    | 6        | 2.80871 | 0.17554    |

### Single GPU Implementation

#### Experimental Results

Increase processes, keeping epochs the same is strong scaling.

Table 4

| Threads per Block | Time (s) | Time Serial (s) | Epochs | Clusters | Speedup |
| ----------------- | -------- | --------------- | ------ | -------- | ------- |
| 256               | 1.726700 | 9.633755        | 50     | 3        | 5.57929 |
| 512               | 1.678286 | 9.381626        | 50     | 3        | 5.59000 |
| 1024              | 1.838031 | 9.616937        | 50     | 3        | 5.23220 |
| 256               | 1.731712 | 12.514634       | 50     | 4        | 7.22674 |
| 512               | 1.718942 | 12.332335       | 50     | 4        | 7.17438 |
| 1024              | 1.652933 | 12.598727       | 50     | 4        | 7.62204 |
| 256               | 1.838919 | 18.889932       | 50     | 6        | 10.2723 |
| 512               | 1.670835 | 19.385700       | 50     | 6        | 11.6024 |
| 1024              | 1.717685 | 18.611738       | 50     | 6        | 10.8354 |

Increase epochs and threads proportionally is weak scaling.

Table 5

| Threads per Block | Time (s) | Time Serial (s) | Epochs | Clusters | Speedup |
| ----------------- | -------- | --------------- | ------ | -------- | ------- |
| 256               | 1.726700 | 9.633755        | 50     | 3        | 5.57929 |
| 512               | 3.224774 | 19.231200       | 100    | 3        | 5.96358 |
| 1024              | 6.292316 | 38.413889       | 200    | 3        | 6.10489 |
| 256               | 1.731712 | 12.514634       | 50     | 4        | 7.22674 |
| 512               | 3.190473 | 24.911140       | 100    | 4        | 7.80798 |
| 1024              | 6.146267 | 50.652925       | 200    | 4        | 8.24125 |
| 256               | 1.838919 | 18.889932       | 50     | 6        | 10.2723 |
| 512               | 3.311074 | 37.237659       | 100    | 6        | 11.2464 |
| 1024              | 6.135140 | 75.330369       | 200    | 6        | 12.2785 |

## Analysis of Distributed Implementations

### Distributed CPU Implementation

On 3 clusters:

Table 6

| Nodes | Time (s) | Time Serial (s) | Epochs | Clusters |
| ----- | -------- | --------------- | ------ | -------- |
| 2     | 0.766036 | 4.811508        | 25     | 3        |
| 3     | 0.621373 | 4.827385        | 25     | 3        |
| 4     | 0.581396 | 4.842869        | 25     | 3        |

On 4 clusters:

Table 7

| Nodes | Time (s) | Time Serial (s) | Epochs | Clusters |
| ----- | -------- | --------------- | ------ | -------- |
| 2     | 0.947897 | 6.203727        | 25     | 4        |
| 3     | 0.739652 | 6.202104        | 25     | 4        |
| 4     | 0.670900 | 6.228862        | 25     | 4        |

On 6 Clusters:

Table 8

| Nodes | Parallel Time (s) | Time Serial (s) | Epochs | Clusters |
| ----- | ----------------- | --------------- | ------ | -------- |
| 2     | 1.245606          | 8.818218        | 25     | 6        |
| 3     | 0.942043          | 8.863819        | 25     | 6        |
| 4     | 0.819143          | 8.859402        | 25     | 6        |

Notice how the parallel time is going down as we increase the number of nodes. This breaks up the amount of data to process per node and allows us to process the data faster.

Now with 4 nodes but scaling up the number of epochs and the amount of data:

Table 9

| Nodes | Parallel Time (s) | Time Serial (s) | Epochs | Clusters |
| ----- | ----------------- | --------------- | ------ | -------- |
| 4     | 0.820076          | 8.816901        | 25     | 6        |
| 4     | 1.595893          | 17.698479       | 50     | 6        |
| 4     | 3.121529          | 35.421519       | 100    | 6        |
| 4     | 6.180246          | 70.824262       | 200    | 6        |

### Distributed GPU Implementation

On 3 Clusters:
Table 10

| Nodes | Parallel Time (s) | Time Serial (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 2     | 0.707328          | 4.711447        | 25     | 3        | 256               |
| 3     | 0.647141          | 4.714513        | 25     | 3        | 256               |
| 4     | 0.799928          | 4.704620        | 25     | 3        | 256               |

On 4 Clusters:
Table 11

| Nodes | Parallel Time (s) | Time Serial (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 2     | 0.634951          | 6.065612        | 25     | 4        | 256               |
| 3     | 0.814948          | 6.070404        | 25     | 4        | 256               |
| 4     | 0.804700          | 6.096105        | 25     | 4        | 256               |

On 6 Clusters:
Table 12

| Nodes | Parallel Time (s) | Time Serial (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 2     | 0.778927          | 8.686541        | 25     | 6        | 256               |
| 3     | 0.657219          | 8.690562        | 25     | 6        | 256               |
| 4     | 0.821848          | 8.679386        | 25     | 6        | 256               |

Here we see that as we increase the number of nodes the time does go down for 3, but jumps back up to 4. This could be due to the amount of overhead involved in running this on only 25 epochs. As we increase the number of epochs from 25 to 100 we see much clearer the advantage of using more nodes.

To illustrate the point, here are the results for 100 epochs and 6 clusters:
Table 13

| Nodes | Parallel Time (s) | Time Serial (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 2     | 1.847004          | 34.788968       | 100    | 6        | 256               |
| 3     | 1.797190          | 34.751091       | 100    | 6        | 256               |
| 4     | 2.352228          | 34.801689       | 100    | 6        | 256               |

<!-- TODO: Landon Write discussion about how the distributed CPU and distributed GPU compare -->

Now with 4 nodes but scaling up the number of epochs
Table 15

| Nodes | Parallel Time (s) | Time Serial (s) | Epochs | Clusters | Threads per Block |
| ----- | ----------------- | --------------- | ------ | -------- | ----------------- |
| 4     | 0.821734          | 8.690958        | 25     | 6        | 256               |
| 4     | 1.296684          | 17.392156       | 50     | 6        | 256               |
| 4     | 2.257343          | 34.808612       | 100    | 6        | 256               |
| 4     | 4.277128          | 69.621106       | 200    | 6        | 256               |

On this table, we see that the increasing epochs does increase the time, and doubling the amount of data very nearly doubles the time. This is to be expected.

100 epochs and 6 clusters and same number of nodes and different threads per block:
Table 16

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
