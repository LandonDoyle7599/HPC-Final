## Analysis of Parallel Implementations

### Serial Implementation - Done?

| Time (s) | Epochs | Clusters |
| -------- | ------ | -------- |
| 8.185362 | 25     | 3        |
| 32.68488 | 100    | 3        |
| 62.56634 | 100    | 6        |
| 117.9167 | 100    | 12       |
| 65.36135 | 200    | 3        |
| 125.1088 | 200    | 6        |
| 235.8639 | 200    | 12       |
| 130.7151 | 400    | 3        |
| 250.3113 | 400    | 6        |
| 471.7810 | 400    | 12       |

### Parallel CPU Implementation

Increase processes, keeping epochs the same is strong scaling.

| Threads | Time (s) | Epochs | Clusters | Speedup | Efficiency |
| ------- | -------- | ------ | -------- | ------- | ---------- |
| 4       | 10.95695 | 100    | 3        | 2.98303 | 0.74576    |
| 8       | 9.345361 | 100    | 3        | 3.49744 | 0.43718    |
| 16      | 6.706964 | 100    | 3        | 4.87328 | 0.30458    |
| 4       | 18.43673 | 100    | 6        | 3.39357 | 0.84839    |
| 8       | 14.02625 | 100    | 6        | 4.46066 | 0.55758    |
| 16      | 9.576324 | 100    | 6        | 6.53344 | 0.40834    |
| 4       | 32.28401 | 100    | 12       | 3.65248 | 0.91312    |
| 8       | 21.50183 | 100    | 12       | 5.48403 | 0.68550    |
| 16      | 15.09110 | 100    | 12       | 7.81366 | 0.48835    |

Increase epochs and threads proportionally is weak scaling.

| Threads | Time (s)  | Epochs | Clusters | Speedup | Efficiency |
| ------- | --------- | ------ | -------- | ------- | ---------- |
| 4       | 10.95695  | 100    | 3        | 2.98303 | 0.74576    |
| 8       | 18.52852  | 200    | 3        | 3.52760 | 0.44095    |
| 16      | 26.73639  | 400    | 3        | 4.88903 | 0.30556    |
| 4       | 18.43673  | 100    | 6        | 3.39357 | 0.84839    |
| 8       | 29.61753  | 200    | 6        | 4.22415 | 0.52802    |
| 16      | 30.336343 | 400    | 6        |         |            |
| 4       | 32.28401  | 100    | 12       | 3.65248 | 0.91312    |
| 8       | 28.141354 | 200    | 12       |         |            |
| 16      | 30.336343 | 400    | 12       |         |            |

### Single GPU Implementation

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

### Distributed GPU Implementation

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
