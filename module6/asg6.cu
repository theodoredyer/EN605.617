/*
Problem Spec: 
- Create program that utilizes register memory. 
- Utilize register variables, preferably executing a similar algorithm with shared or other types of memory. 
- Any comparison of timing, if the kernel code is the same, will earn extra points.

Theodore Dyer
Introduction to GPU Programming Spring 2022 (EN605.617.81)
Chance Pascale
3/14/2020

*/

#include <stdio.h>
#include <stdlib.h>

#define KERNEL_LOOP 1028
#define KERNEL_SIZE 32

__constant__ static const int const_data_test = 2;


// from global_memory.cu file provided in Module 4 Vocareum lab
// create a timer object to test the duration of an operation
__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}


/*
generate_data()
- generates data to be later used in testing utilization of register variables.

- Params:
    host_data_ptr = pointer to array in host memory that will be filled with 
        values through execution of this function

- Return:
    Void, however upon return the array pointer to by
    'host_data_ptr' will be populated with values
*/
__host__ void generate_data(int * host_data_ptr)
{
        for(int i=0; i < KERNEL_LOOP; i++)
        {
                host_data_ptr[i] = i;
        }
}


/*
const_register_add()
- tests the utilization of adding constant and register variables

- Params:
    data = input data upon which we will execute operations, where intermediate values will be 
        held in register variables
    num_elements = tracker for the number of elements we are seeking to process, note if this value
        exceeds the number of registers available to us we will no longer be utilizing register variables. 

- Return:
    Void, however upon return data[x] = data[x] + const_data_test
*/
__global__ void const_register_test(int * const data, const int num_elements)
{
        const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        printf("tid = %d\n", tid);

        // Note - if tid > num_elements we are likely in a scenario where we 
        // do not have adequate registers to allocate to each thread either way
        if(tid < num_elements)
        {
                // If we have enough registers available to be allocated, 
                // d_tmp will be stored in a register. 
                int d_tmp = data[tid];
                d_tmp = d_tmp + const_data_test;
                d_tmp = d_tmp * const_data_test;
                data[tid] = d_tmp;
        }
}

/*
no_const_register_add()
- tests the utilization of register variables by storing and using single values

- Params:
    data = input data upon which we will execute operations, where intermediate values will be 
        held in register variables
    num_elements = tracker for the number of elements we are seeking to process, note if this value
        exceeds the number of registers available to us we will no longer be utilizing register variables. 

- Return:
    Void, however upon return data[i] = data[i] + 1
*/
__global__ void no_const_register_test(int * const data, const int num_elements)
{
        const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

        // Note - if tid > num_elements we are likely in a scenario where we 
        // do not have adequate registers to allocate to each thread either way
        if(tid < num_elements)
        {
                // If we have enough registers available to be allocated, 
                // d_tmp will be stored in a register. 
                int d_tmp = data[tid];
                d_tmp = d_tmp + 2;
                d_tmp = d_tmp - 1;
                data[tid] = d_tmp;
        }
}

__host__ void gpu_register_test(int modifier)
{
        const int num_elements = KERNEL_LOOP / modifier;
        const int num_threads = KERNEL_SIZE;
        const int num_blocks = (num_elements + num_threads - 1)/num_threads;
        const int num_bytes = num_elements * sizeof(int);

        int * d_gpu;
        int * dc_gpu;

        int i_host_data[num_elements];
        int o_host_data[num_elements];

        generate_data(i_host_data);

        // #########################################################################################
        // Start - Pure register variable test.
        
        printf("\nStarting pure register variable test...\n");

        // Allocate memory and send data to device
        cudaMalloc(&d_gpu, num_bytes);

        cudaEvent_t start_time = get_time();
        cudaMemcpy(d_gpu, i_host_data, num_bytes, cudaMemcpyHostToDevice);

        // Execute kernel operation
        no_const_register_test <<<num_blocks, num_threads>>>(d_gpu, num_elements);

        // Wait for the GPU launched work to complete
        cudaThreadSynchronize();        
        cudaGetLastError();

        // Return data from device back to host
        cudaMemcpy(o_host_data, d_gpu, num_bytes,cudaMemcpyDeviceToHost);

        printf("\nOutput should be (input + 1)\n");
        for (int i = 0; i < num_elements; i++){
                printf("Input value: %d, output: %d\n", i_host_data[i], o_host_data[i]);
        }

        // finish timing performance and record result
        cudaEvent_t end_time = get_time();
        cudaEventSynchronize(end_time);
        float pure = 0;
        cudaEventElapsedTime(&pure, start_time, end_time);
        printf("Time from allocation to completion for pure register variable test: %f \n", pure);

        cudaFree((void* ) d_gpu);
        cudaDeviceReset();
         
        // End - Pure register variable test.
        // #########################################################################################


        // #########################################################################################
        // Start - Register / Constant memory test

        // Allocate memory and send data to device
        cudaMalloc(&dc_gpu, num_bytes);

        printf("\nStarting register variable + constant memory test...\n");
        cudaEvent_t const_start_time = get_time();
        cudaMemcpy(dc_gpu, i_host_data, num_bytes, cudaMemcpyHostToDevice);

        // Execute kernel operation
        const_register_test <<<num_blocks, num_threads>>>(dc_gpu, num_elements);

        // Wait for the GPU launched work to complete
        cudaThreadSynchronize();        
        cudaGetLastError();

        // Return data from device back to host
        cudaMemcpy(o_host_data, dc_gpu, num_bytes, cudaMemcpyDeviceToHost);

        printf("\nOutput should be = (input + 2) * 2\n");
        for (int i = 0; i < num_elements; i++){
                printf("Input value: %d, output: %d\n", i_host_data[i], o_host_data[i]);
        }

        // finish timing performance and record result
        end_time = get_time();
        cudaEventSynchronize(end_time);
        float delta = 0;
        cudaEventElapsedTime(&delta, const_start_time, end_time);

        printf("\nResults for test size = %d", num_elements);
        printf("\nTime from allocation to completion for pure register variable test: %f \n", pure);
        printf("Time from allocation to completion for constant memory / register variable test: %f \n", delta);

        cudaFree((void* ) dc_gpu);
        cudaDeviceReset();
         
        // End - Pure register variable test.
        // #########################################################################################
        
}

void execute_host_functions()
{

}

void execute_gpu_functions()
{
	gpu_register_test(1);
}


int main(void) {
	execute_host_functions();
	execute_gpu_functions();

	return EXIT_SUCCESS;
}
