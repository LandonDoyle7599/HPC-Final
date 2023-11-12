
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<unistd.h>
#include<math.h>
#include<errno.h>
#include<mpi.h>


int numEpochs = 0;
int numCentroids = 0;
int numElements = 0;
int numProcesses = 0;

/* This function goes through that data points and assigns them to a cluster */
void assign2Cluster(double k_x[], double k_y[], double recv_x[], double recv_y[], int assign[])
{
	double min_dist = 10000000;
	double x=0, y=0, temp_dist=0;
	int k_min_index = 0;

	for(int i = 0; i < (numElements/numProcesses) + 1; i++)
	{
		for(int j = 0; j < numCentroids; j++)
		{
			x = abs(recv_x[i] - k_x[j]);
			y = abs(recv_y[i] - k_y[j]);
			temp_dist = (x*x) + (y*y);

			// new minimum distance found
			if(temp_dist < min_dist)
			{
				min_dist = temp_dist;
				k_min_index = j;
			}
		}

		// update the cluster assignment of this data points
		assign[i] = k_min_index;
	}

}

/* Recalcuate k-means of each cluster because each data point may have
   been reassigned to a new cluster for each iteration of the algorithm */
void calcKmeans(double k_means_x[], double k_means_y[], double data_x_points[], double data_y_points[], int k_assignment[])
{
	double total_x = 0;
	double total_y = 0;
	int numOfpoints = 0;

	for(int i = 0; i < numCentroids; i++)
	{
		total_x = 0;
		total_y = 0;
		numOfpoints = 0;

		for(int j = 0; j < numElements; j++)
		{
			if(k_assignment[j] == i)
			{
				total_x += data_x_points[j];
				total_y += data_y_points[j];
				numOfpoints++;
			}
		}

		if(numOfpoints != 0)
		{
			k_means_x[i] = total_x / numOfpoints;
			k_means_y[i] = total_y / numOfpoints;
		}
	}

}

int main(int argc, char *argv[])
{
	// initialize the MPI environment
	MPI_Init(NULL, NULL);

	// get number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// get rank
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// send buffers
	double *k_means_x = NULL;		// k means corresponding x values
	double *k_means_y = NULL;		// k means corresponding y values
	int *k_assignment = NULL;		// each data point is assigned to a cluster
	double *data_x_points = NULL;
	double *data_y_points = NULL;

	// receive buffer
	double *recv_x = NULL;
	double *recv_y = NULL;
	int *recv_assign = NULL;

	if(world_rank == 0)
	{
		if(argc != 3)
		{
			printf("Please include an argument after the program name to list how many processes.\n");
			printf("e.g. To indicate 4 processes, run: mpirun -n 4 ./kmeans 4\n");
			exit(-1);
		}

		char buffer[2];

        // Get clusters and epochs from command line
        numEpochs = atoi(argv[1]);
        numCentroids = atoi(argv[2]);

        numProcesses = world_size;

		// broadcast the number of clusters to all nodes
		MPI_Bcast(&numCentroids, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numEpochs, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numProcesses, 1, MPI_INT, 0, MPI_COMM_WORLD);


		// allocate memory for arrays
		k_means_x = (double *)malloc(sizeof(double) * numCentroids);
		k_means_y = (double *)malloc(sizeof(double) * numCentroids);

		if(k_means_x == NULL || k_means_y == NULL)
		{
			perror("malloc");
			exit(-1);
		}

        // TODO: Change this to read in data from song_data.csv
        // For loop to randomly create data points
        numElements = 1000000;

		// broadcast the number of elements to all nodes
		MPI_Bcast(&numElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// allocate memory for an array of data points
		data_x_points = (double *)malloc(sizeof(double) * numElements);
		data_y_points = (double *)malloc(sizeof(double) * numElements);
		k_assignment = (int *)malloc(sizeof(int) * numElements);

		if(data_x_points == NULL || data_y_points == NULL || k_assignment == NULL)
		{
			perror("malloc");
			exit(-1);
		}


		// now fill the arrays
		int i = 0;

		double point_x=0, point_y=0;

		while(i < numElements)
        {
            point_x = (double)rand();
            point_y = (double)rand();

            data_x_points[i] = point_x;
            data_y_points[i] = point_y;

            // assign the initial k means to zero
            k_assignment[i] = 0;
            i++;
        }

		// randomly select initial k-means
		time_t t;
		srand((unsigned) time(&t));
		int random;
		for(int i = 0; i < numCentroids; i++) {
			random = rand() % numElements;
			k_means_x[i] = data_x_points[random];
			k_means_y[i] = data_y_points[random];
		}

		printf("Running k-means algorithm for %d iterations...\n\n", numEpochs);
		for(int i = 0; i < numCentroids; i++)
		{
			printf("Initial K-means: (%f, %f)\n", k_means_x[i], k_means_y[i]);
		}

		// allocate memory for receive buffers
        printf("Number of processes: %d\n", numProcesses);
		recv_x = (double *)malloc(sizeof(double) * ((numElements/numProcesses) + 1));
		recv_y = (double *)malloc(sizeof(double) * ((numElements/numProcesses) + 1));
		recv_assign = (int *)malloc(sizeof(int) * ((numElements/numProcesses) + 1));

		if(recv_x == NULL || recv_y == NULL || recv_assign == NULL)
		{
			perror("malloc");
			exit(-1);
		}
	}
	else
	{	// Worker Node

		// receive broadcast of number of clusters
		MPI_Bcast(&numCentroids, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// receive broadcast of number of elements
		MPI_Bcast(&numElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // receive broadcast of number of epochs
        MPI_Bcast(&numEpochs, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // receive broadcast of number of processes
        MPI_Bcast(&numProcesses, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// allocate memory for arrays
		k_means_x = (double *)malloc(sizeof(double) * numCentroids);
		k_means_y = (double *)malloc(sizeof(double) * numCentroids);

		if(k_means_x == NULL || k_means_y == NULL)
		{
			perror("malloc");
			exit(-1);
		}

		// allocate memory for receive buffers
		recv_x = (double *)malloc(sizeof(double) * ((numElements/numProcesses) + 1));
		recv_y = (double *)malloc(sizeof(double) * ((numElements/numProcesses) + 1));
		recv_assign = (int *)malloc(sizeof(int) * ((numElements/numProcesses) + 1));

		if(recv_x == NULL || recv_y == NULL || recv_assign == NULL)
		{
			perror("malloc");
			exit(-1);
		}
	}

	/* Distribute the work among all nodes. The data points itself will stay constant and
	   not change for the duration of the algorithm. */
	MPI_Scatter(data_x_points, (numElements/numProcesses) + 1, MPI_DOUBLE,
		recv_x, (numElements/numProcesses) + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Scatter(data_y_points, (numElements/numProcesses) + 1, MPI_DOUBLE,
		recv_y, (numElements/numProcesses) + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	int epoch = 0;
	while(epoch < numEpochs)
	{
		// broadcast k-means arrays
		MPI_Bcast(k_means_x, numCentroids, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(k_means_y, numCentroids, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// scatter k-cluster assignments array
		MPI_Scatter(k_assignment, (numElements/numProcesses) + 1, MPI_INT,
			recv_assign, (numElements/numProcesses) + 1, MPI_INT, 0, MPI_COMM_WORLD);

		// assign the data points to a cluster
		assign2Cluster(k_means_x, k_means_y, recv_x, recv_y, recv_assign);

		// gather back k-cluster assignments
		MPI_Gather(recv_assign, (numElements/numProcesses)+1, MPI_INT,
			k_assignment, (numElements/numProcesses)+1, MPI_INT, 0, MPI_COMM_WORLD);

		// let the root process recalculate k means
		if(world_rank == 0)
		{
			calcKmeans(k_means_x, k_means_y, data_x_points, data_y_points, k_assignment);
			//printf("Finished iteration %d\n",epoch);
		}

		epoch++;
	}

	if(world_rank == 0)
	{
		printf("--------------------------------------------------\n");
		printf("FINAL RESULTS:\n");
		for(int i = 0; i < numCentroids; i++)
		{
			printf("Cluster #%d: (%f, %f)\n", i, k_means_x[i], k_means_y[i]);
		}
		printf("--------------------------------------------------\n");
	}

	// deallocate memory and clean up
	free(k_means_x);
	free(k_means_y);
	free(data_x_points);
	free(data_y_points);
	free(k_assignment);
	free(recv_x);
	free(recv_y);
	free(recv_assign);

	//MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

}