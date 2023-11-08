# Spotify Visualizer

## How to Run

### Serial CPU

We used a Cmake file to compile our project. To run the project, you must have Cmake and a c++ compiler.

To run the project, you must first clone the repository. Then, you must run the following commands in the terminal:

```bash
g++ main.cpp -o main
./main
```

### Parallel CPU

### Distributed CPU

### Parallel GPU

Running on CHPC first we need to load the module:

```bash
module load cuda/12
```

Now we can compile and execute:

```bash
nvcc gpu.cu -o gpu
./gpu
```

### Distributed GPU

## Validation

In serial.hpp we wrote a function, areFilesEqual, to validate two csv files against eachother. It will return true if the are, false if not. We will check every file against the ground truth, defined by the serial implementation.

## To Run the Python Visualization

Run the following commands from the project root directory:

```bash
pip install -r requirements.txt
python3 visualize.py
```

## Analysis

TODO: Insert Images and table comparing times, epochs, and nodes here.

### Serial Implementation

|NumPoints|Time (s)|Epochs| Clusters|
|----|----|------|--|
| 1240425 | 25.816402 |  100 | 6 |
| 1240425 | 48.143714 |  100 | 12|
| 1240425 | 52.172065 |  200 | 6 |
| 1240425 | 95.354178 |  200 | 12 |
