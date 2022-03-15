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

#define sizeOfArray 1024*1024*64


/*
arr_add()
- add two arrays of integers

- Params:
    da = input data array 1, specifies one of the addition operators
    db = input data array 2, specifies the second addition operator
    dr = array into which we will populate values of the addition

- Return:
    void, but upon return dr[i] = da[i] + db[i]
*/
__global__ void arr_add(int *da, int *db, int *dr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < sizeOfArray) {
        dr[tid] = da[tid] + db[tid];
    }
}

/*
arr_sub()
- subtracts two arrays of integers

- Params:
    da = input data array 1, the array that will be subtracted from
    db = input data array 2, values will be decremented from da
    dr = array into which we will populate values of the subtraction

- Return:
    void, but upon return dr[i] = da[i] - db[i]
*/
__global__ void arr_sub(int *da, int *db, int *dr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < sizeOfArray) {
        dr[tid] = da[tid] - db[tid];
    }
}

/*
arr_mul()
- multiplies two arrays of integers

- Params:
    da = input data array 1, specifies multiplication operator 1
    db = input data array 2, specifies multiplication operator 2
    dr = array into which we will populate values of the multiplication

- Return:
    void, but upon return dr[i] = da[i] * db[i]
*/
__global__ void arr_mul(int *da, int *db, int *dr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < sizeOfArray) {
        dr[tid] = da[tid] * db[tid];
    }
}

/*
arr_div()
- multiplies two arrays of integers

- Params:
    da = input data array 1, specifies multiplication operator 1
    db = input data array 2, specifies multiplication operator 2
    dr = array into which we will populate values of the multiplication

- Return:
    void, but upon return dr[i] = da[i] * db[i]
*/
__global__ void arr_div(int *da, int *db, int *dr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < sizeOfArray) {
        dr[tid] = da[tid] / db[tid];
    }
}


/*
run_test()
- runs a timed test utilizing CUDA streams/events

- Params:
    operation = string specifying which type of test to run, can be one of 4 mathematical
        operations ("addition", "subtraction", "multiplication", or "division")

- Return:
    essentially void, results of the test will be printed upon execution - keeping int for 
        now for debugging purposes being able to use return 0 as an escape. 
*/
int run_test(const char* operation) {

    printf("--------------------------------------------------\n");
    printf("\n Running test, math operation = %s\n", operation);

    cudaDeviceProp prop;
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
        host_a[i] = ((i % 50) * 2) + 2;
        host_b[i] = (((i % 50) * 2) + 2)/2;
    }

    cudaEventRecord(start);

    // Set up data copy to device
    cudaMemcpyAsync(device_a, host_a, sizeOfArray * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_b, host_b, sizeOfArray * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Kernel
    if(operation == "addition") {
        arr_add<<<sizeOfArray, 1, 1, stream>>>(device_a, device_b, device_r);
    } else if(operation == "subtraction"){
        arr_sub<<<sizeOfArray, 1, 1, stream>>>(device_a, device_b, device_r);
    } else if(operation == "multiplication") {
        arr_mul<<<sizeOfArray, 1, 1, stream>>>(device_a, device_b, device_r);
    } else if(operation == "division"){
        arr_div<<<sizeOfArray, 1, 1, stream>>>(device_a, device_b, device_r);
    }
    
    
    cudaMemcpyAsync(host_r, device_r, sizeOfArray * sizeof(int), cudaMemcpyDeviceToHost, stream);

    // for(int i = 0; i < sizeOfArray; i++) {
    //     printf("a = %d, b = %d, r = %d\n", host_a[i], host_b[i], host_r[i]);
    // }

    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("\n Size of array : %d \n", sizeOfArray);
    printf("\n Math operation (%s) Time taken: %3.1f ms\n\n", operation, elapsed_time);
    printf("--------------------------------------------------\n");

    // Free Allocated Memory
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_r);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_r);

    return 0;

}

int main() {

    run_test("addition");
    run_test("subtraction");
    run_test("multiplication");
    run_test("division");
    
    return EXIT_SUCCESS;
}
