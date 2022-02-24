#include <stdio.h>

/*

Problem Spec: Create a program that utilizes a variety of combinations
of large threads executing over blocks based of various sizes.

GPU Programming Spring 2022 - EN605.617.81
Theodore Dyer
Chance Pascale
Module 3 Assignment

*/


// Note - I'm using the grids.cu provided file as a starting point for this assignment. 

// Data size = size_x * size_y
#define DATA_SIZE_X 64
#define DATA_SIZE_Y 32
#define DATA_SIZE_IN_BYTES ((DATA_SIZE_X) * (DATA_SIZE_Y) * (sizeof(unsigned int)))

__global__
void get_id_2d_data(
    unsigned int * const block_x,
    unsigned int * const block_y,
    unsigned int * const thread,
    unsigned int * const calc_thread,
    unsigned int * const thread_x,
    unsigned int * const thread_y,
    unsigned int * const grid_dim_x,
    unsigned int * const grid_dim_y,
    unsigned int * const block_dim_x,
    unsigned int * const block_dim_y) {

    const unsigned int id_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int id_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    const unsigned int thread_index = ((gridDim.x * blockDim.x) * id_y) + id_x;
    
    block_x[thread_index] = blockIdx.x;
    block_y[thread_index] = blockIdx.y;
    thread[thread_index] = threadIdx.x;
    calc_thread[thread_index] = thread_index;
    thread_x[thread_index] = id_x;
    thread_y[thread_index] = id_y;
    grid_dim_x[thread_index] = gridDim.x;
    grid_dim_y[thread_index] = gridDim.y;
    block_dim_x[thread_index] = blockDim.x;
    block_dim_y[thread_index] = blockDim.y;

}

unsigned int cpu_block_x[DATA_SIZE_Y][DATA_SIZE_X];
unsigned int cpu_block_y[DATA_SIZE_Y][DATA_SIZE_X];
unsigned int cpu_thread[DATA_SIZE_Y][DATA_SIZE_X];
unsigned int cpu_warp[DATA_SIZE_Y][DATA_SIZE_X];
unsigned int cpu_calc_thread[DATA_SIZE_Y][DATA_SIZE_X];
unsigned int cpu_thread_x[DATA_SIZE_Y][DATA_SIZE_X];
unsigned int cpu_thread_y[DATA_SIZE_Y][DATA_SIZE_X];
unsigned int cpu_grid_dim_x[DATA_SIZE_Y][DATA_SIZE_X];
unsigned int cpu_grid_dim_y[DATA_SIZE_Y][DATA_SIZE_X];
unsigned int cpu_block_dim_x[DATA_SIZE_Y][DATA_SIZE_X];
unsigned int cpu_block_dim_y[DATA_SIZE_Y][DATA_SIZE_X];

int main(void) {

    /* Thread count * block count (should) = data size */
    // Note data is 64 * 32 = 2048

    /*
        Test One = flat rectangle, 1 x 8 grid of 32 x 8 blocks
        Total thread count = 32 * 8 = 256
    */
    const dim3 threads_test_one(32,8);
    const dim3 blocks_test_one(1,8);

    /*
        Test One = Rectangle 2, 2 x 4 grid of 16 x 16 blocks
        Total thread count = 16 * 16 = 256
    */
    const dim3 threads_test_two(16,16);
    const dim3 blocks_test_two(2,4);

    /*
        Test Three = Rectangle 3, 4 x 2 grid of 64 x 4 blocks
        Total thread count = 64 * 4 = 256
    */
    const dim3 threads_test_three(64,4);
    const dim3 blocks_test_three(4,2);

    char ch;

    unsigned int * gpu_block_x;
	unsigned int * gpu_block_y;
	unsigned int * gpu_thread;
	unsigned int * gpu_warp;
	unsigned int * gpu_calc_thread;
	unsigned int * gpu_thread_x;
	unsigned int * gpu_thread_y;
	unsigned int * gpu_grid_dim_x;
    unsigned int * gpu_grid_dim_y;
    unsigned int * gpu_block_dim_x;
    unsigned int * gpu_block_dim_y;
	
    // Allocate arrays on GPU
    cudaMalloc((void **)&gpu_block_x, DATA_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_block_y, DATA_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_thread, DATA_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_warp, DATA_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_calc_thread, DATA_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_thread_x, DATA_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_thread_y, DATA_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_grid_dim_x, DATA_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_grid_dim_y, DATA_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_block_dim_x, DATA_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_block_dim_y, DATA_SIZE_IN_BYTES);

    // Executing...
    for(int kernel = 0; kernel < 3; kernel++) {
        switch(kernel) {
            case 0:
            {
                // Execute kernel
                get_id_2d_data<<<blocks_test_one, threads_test_one>>>(gpu_block_x, gpu_block_y, gpu_thread, gpu_calc_thread, gpu_thread_x, gpu_thread_y, gpu_grid_dim_x, gpu_grid_dim_y, gpu_block_dim_x, gpu_block_dim_y);
            } break;

            case 1:
            {
                get_id_2d_data<<<blocks_test_two, threads_test_two>>>(gpu_block_x, gpu_block_y, gpu_thread, gpu_calc_thread, gpu_thread_x, gpu_thread_y, gpu_grid_dim_x, gpu_grid_dim_y, gpu_block_dim_x, gpu_block_dim_y);
            } break;

            case 2:
            {
                get_id_2d_data<<<blocks_test_three, threads_test_three>>>(gpu_block_x, gpu_block_y, gpu_thread, gpu_calc_thread, gpu_thread_x, gpu_thread_y, gpu_grid_dim_x, gpu_grid_dim_y, gpu_block_dim_x, gpu_block_dim_y);
            } break;

            default: exit(1); break;
        }

        /* Bring GPU results to CPU */
		cudaMemcpy(cpu_block_x, gpu_block_x, DATA_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_block_y, gpu_block_y, DATA_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_thread, gpu_thread, DATA_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_calc_thread, gpu_calc_thread, DATA_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_thread_x, gpu_thread_x, DATA_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_thread_y, gpu_thread_y, DATA_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_grid_dim_x, gpu_grid_dim_x, DATA_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_grid_dim_y, gpu_grid_dim_y, DATA_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_block_dim_x, gpu_block_dim_x, DATA_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_block_dim_y, gpu_block_dim_y, DATA_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

        // Printing output...
        printf("\nKernel %d\n", kernel);
		/* Iterate through the arrays and print */
		for(int y = 0; y < DATA_SIZE_Y; y++)
		{
			for(int x = 0; x < DATA_SIZE_X; x++)
			{
				printf("CT: %2u BKX: %1u BKY: %1u TID: %2u YTID: %2u XTID: %2u GDX: %1u BDX: %1u GDY: %1u BDY: %1u\n",
						cpu_calc_thread[y][x], cpu_block_x[y][x], cpu_block_y[y][x], cpu_thread[y][x], cpu_thread_y[y][x],
						cpu_thread_x[y][x], cpu_grid_dim_x[y][x], cpu_block_dim_x[y][x], cpu_grid_dim_y[y][x], cpu_block_dim_y[y][x]);

			}
		}
    }

    // Free allocated arrays on gpu
	cudaFree(gpu_block_x);
	cudaFree(gpu_block_y);
	cudaFree(gpu_thread);
	cudaFree(gpu_warp);
	cudaFree(gpu_calc_thread);
	cudaFree(gpu_thread_x);
	cudaFree(gpu_thread_y);
	cudaFree(gpu_grid_dim_x);
    cudaFree(gpu_grid_dim_y);
	cudaFree(gpu_block_dim_x);
	cudaFree(gpu_block_dim_y);
}