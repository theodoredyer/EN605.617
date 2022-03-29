/*
Problem Spec: 
- Create a program that utilizes two libraries from all of the libraries in the module. 
- Any comparison of timing, if kernel code is the same, will earn extra points
- Create a program that executes kernel using NVIDIA CUDA toolkit library to execute a simple operation
- Create a program that executes kernel using a second NVIDIA CUDA toolkit library to execute a simple operation
- Test harness executes two separate runs of each kernel
- Output timing or other metrics for comparison of different data sets, by size or other properties

Theodore Dyer
Introduction to GPU Programming Spring 2022 (EN605.617.81)
Chance Pascale
3/28/2020

*/

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>

#include <curand.h>
#include <curand_kernel.h>

#define MAX 100
#define N 25
#define M 20



// Following example in curand_example.cu for setting up curand operations.

// Initializes random states
__global__ void rand_init(unisnged int seed, curandState_t* states) {
    const int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    curand_init(seed, thread_idx, 0, &states[thread_idx]);
}

// Kernel that takes an array of states, array of ints, and puts a random int into each
__global__ void generate_random(curandState_t* states, int* numbers) {
    const int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    numbers[thread_idx] = curand(&states[thread_idx]) % MAX;
}


/*
execute_rand()
- Runs tests associated with the curand library
*/
void execute_rand()
{
    
    printf("--------------------------------------------------\n");
    printf("\n Running random number operation... \n");

    // Initialize time for tracking runtime
    cudaEvent_t start, stop;
    float elapsed_time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Initialize and allocate space for states
    curandState_t* states;
    cudaMalloc((void **) &states, N * M * sizeof(cuRandState_t));

    // Initialize random states via kernel
    rand_init<<<N,M>>>(time(0), states);

    // Allocate an array of ints to be used with states to generate numbers
    int cpu_nums[N*M];
    int* gpu_nums;
    cudaMalloc((void **) &gpu_nums, N * M *sizeof(int));

    // Generate randoms via kernel
    generate_random<<<N,M>>>(states, gpu_nums);

    // Copy data back to host
    cudaMemcpy(cpu_nums, gpu_nums, N * M * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);


    /* print them out */
    printf("Printing out random numbers: \n");
    for (int i = 0; i < N; i++) {
        for (int j =0; j < M; j++){
            printf("%u ", cpu_nums[i * M + j]);
        }
        printf("\n");
    }

    printf("Execution time taken: %3.1f ms\n\n", elapsed_time);

    cudaFree(states);
    cudaFree(gpu_nums);

}

void execute_solve()
{
	
}


int main(void) {
	execute_rand();

	return EXIT_SUCCESS;
}