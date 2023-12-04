For Distributed GPU
```bash
nvcc -c distributedGPU.cu && mpicxx -o dist serial-to-distributed-gpu.cpp distributedGPU.o -L/uufs/chpc.utah.edu/sys/installdir/r8/cuda/12.2.0/lib64 -lcudart -lcuda && mpirun -np 2 ./dist 100 6 && mpirun -np 3 ./dist 100 6 && mpirun -np 4 ./dist 100 6
```

For Serial GPU
```bash
nvcc serial-to-single-gpu.cu -o gpu
./gpu 25 6
```