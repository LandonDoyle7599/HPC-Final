# Spotify Visualizer

## How to Run 

### Serial CPU

We used a Cmake file to compile our project. To run the project, you must have Cmake and a c++ compiler.

To run the project, you must first clone the repository. Then, you must run the following commands in the terminal:

```bash
TODO: Insert Commands Here
```

### Parallel CPU

### Distributed CPU

### Parallel GPU

Running on CHPC first we need to load the module:
```bash
module load cuda/12
```

Now we can compile:

```bash
nvcc gpu.cu -o gpu
```


### Distributed GPU

### To Run the Python Visualization

Run the following commands from the project root directory:

```bash
pip install -r requirements.txt
python3 visualize.py
```

## Analysis

TODO: Insert Images and table comparing times, epochs, and nodes here.
