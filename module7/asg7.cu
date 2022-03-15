/*
Problem Spec: 
- Executes kernel using CUDA streams and events to execute 4 math operations on data that 
    is fed into the kernel from the host code while it is running
- Test harness executes two separate runs of each kernel using CUDA streams and events

Theodore Dyer
Introduction to GPU Programming Spring 2022 (EN605.617.81)
Chance Pascale
3/14/2020

*/

#include <stdio.h> 
#include <time.h>
#include <cuda.h> 

//#define sizeOfArray 1024*1024
#define sizeOfArray 64

__global__ void arr_add(int *da, int *db, int *dr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < sizeOfArray) {
        dr[tid] = da[tid] + db[tid]
    }
}

int main(int argc, char **argv) {
    codaDeviceProp prop;
    int *host_a, *host_b, *host_r;
    int *device_a, *device_b, *device_r;
    int whichDevice;

    cudaGetDeviceCount(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);

    cudaEvent_t start, stop;
    float elapsed_time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Device Allocate
    cudaMalloc((void**)& device_a, sizeOfArray * sizeof(*device_a));
    cudaMalloc((void**)& device_b, sizeOfArray * sizeof(*device_b));
    cudaMalloc((void**)& device_r, sizeOfArray * sizeof(*device_r));

    // Host Allocate
    cudaHostAlloc((void **)&host_a, sizeOfArray*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_b, sizeOfArray*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&host_r, sizeOfArray*sizeof(int), cudaHostAllocDefault);

    // Populate Data
    for(int i = 0; i < sizeOfArray; i++) {
        host_a[i] = rand()%10;
        host_b[i] = rand()%10;
    }

    cudaEventRecord(start);

    // Set up data copy to device
    cudaMemcpyAsync(device_a, host_a, sizeOfArray * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_b, host_b, sizeOfArray * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Kernel
    arr_add<<<sizeOfArray, 1, 1, stream>>>(device_a, device_b, device_r);

    cudaMemcpyAsync(host_r, device_r, sizeOfArray * sizeof(int), cudaMemcpyDeviceToHost, stream);

    for(int i = 0; i < sizeOfArray; i++) {
        printf("hosta = %d, hostb = %d, hostr = %d\n", host_a[i], host_b[i], host_r[i]);
    }

    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("\nResults...\n");
    printf("\n Size of array : %d \n", sizeOfArray);
    printf("\n Time taken: %3.1f ms \n", elapsedTime);

    // Free Allocated Memory
    cudaFreeHost(host_a);
    cudaFreehost(host_b);
    cudaFreeHost(host_result);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_r);

    return 0;

}

/*
function()
- generates data to be later used in testing utilization of register variables.

- Params:
    host_data_ptr = pointer to array in host memory that will be filled with 
        values through execution of this function

- Return:
    Void, however upon return the array pointer to by
    'host_data_ptr' will be populated with values
*/