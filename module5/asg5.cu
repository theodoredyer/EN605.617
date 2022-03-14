/*
Problem Spec: 
- Program that utilizes both shared and constant memory. 
- Utilize both types of device memory, either in one or two separate kernel executions.
- 4 simple math operations, copying data from global to/from shared memory
- 4 simple math operations utilizing shared memory
- 4 simple math operations copying data from global to/from constant memory
- 4 simple math operations utilizing constant memory
- Test harness executes two separate runs of each kernel using shared and constant memory
- Output timing to compare different memory types
*/

/* Theodore Dyer
Introduction to GPU Programming Spring 2022 (EN605.617.81)
Chance Pascale
*/

#include <stdio.h>
#include <stdlib.h>

typedef unsigned short int u16;
typedef unsigned int u32;

#define NUM_ELEMENTS 2048
#define MAX_NUM_LISTS 16
#define KERNEL_LOOP 1024
#define WORK_SIZE 32

#define ARR_LEN 16

__constant__ static const int const_data_start = 100;
__constant__ static const int const_data_mult = 3;
__constant__ static const int const_data_div = 25;



// #########################################################################################
// Start - Shared Memory Functions

/*
staticSharedAdd()
- implements basic addition of two arrays utilizing shared memory

- Params:
    d = input data array 1
    d2 = input data array 2
    len = length of input array

- Return:
    Void, however upon return the array d will hold the result
    of addition of d and d2 via using shared s[]
*/
__global__ void staticSharedAdd(int *d, int *d2, int len) {

    __shared__ int s[ARR_LEN];
    int t = threadIdx.x;
    s[t] = d[t] + d2[t];
    __syncthreads();
    d[t] = s[t];
}

/*
staticSharedMult()
- implements multiplication of array elements by 2 using shared memory

- Params:
    d = input data array 1
    len = length of input array

- Return:
    Void, however upon return the array d will hold values d * 2
*/
__global__ void staticSharedMult(int *d, int len) {

    __shared__ int s[ARR_LEN];
    int t = threadIdx.x;
    s[t] = d[t] * 2;
    __syncthreads();
    d[t] = s[t];
}

/*
staticSharedSub()
- implements basic subtraction of two arrays utilizing shared memory

- Params:
    d = input data array 1
    d2 = input data array 2
    len = length of input array

- Return:
    Void, however upon return the array d will hold the result of
    dubtracting d2 from d via using shared s[]
*/
__global__ void staticSharedSub(int *d, int *d2, int len) {

    __shared__ int s[ARR_LEN];
    int t = threadIdx.x;
    s[t] = d[t] - d2[t];
    __syncthreads();
    d[t] = s[t];
}

/*
staticSharedDiv()
- implements basic division of two arrays utilizing shared memory

- Params:
    d = input data array 1
    d2 = input data array 2
    len = length of input array

- Return:
    Void, however upon return the array d will hold the result of
    dubtracting dividing d by d2 using shared s[]
*/
__global__ void staticSharedDiv(int *d, int *d2, int len) {

    __shared__ int s[ARR_LEN];
    int t = threadIdx.x;
    s[t] = d[t] / d2[t];
    __syncthreads();
    d[t] = s[t];
}
// End - Shared Memory Functions
// #########################################################################################

// #########################################################################################
// Start - Constant Memory Functions

/*
const_test_gpu_const_add()
- implements addition utilizing constant memory

- Params:
    data = input data array of integers
    num_elements = number of elements in the input array

- Return:
    Void, however upon return the array 'data' will hold the
    result of adding data[i] + const_data_start
*/
__global__ void const_test_gpu_const_add(int * const data, const int num_elements) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid < num_elements) {
        int d = const_data_start;
        d += data[tid];
        data[tid] = d;
    }
}

/*
const_test_gpu_const_sub()
- implements subtraction utilizing constant memory

- Params:
    data = input data array of integers
    num_elements = number of elements in the input array

- Return:
    Void, however upon return the array 'data' will hold the
    result of subtracting data[i] from const_data_start
*/
__global__ void const_test_gpu_const_sub(int * const data, const int num_elements) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid < num_elements) {
        int d = const_data_start;
        d -= data[tid];
        data[tid] = d;
    }
}

/*
const_test_gpu_const_mult()
- implements multiplication utilizing constant memory

- Params:
    data = input data array of integers
    num_elements = number of elements in the input array

- Return:
    Void, however upon return the array 'data' will hold the
    result of multiplying data[i] by const_data_mult
*/
__global__ void const_test_gpu_const_mult(int * const data, const int num_elements) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid < num_elements) {
        int d = const_data_mult;
        d *= data[tid];
        data[tid] = d;
    }
}

/*
const_test_gpu_const_div()
- implements division utilizing constant memory

- Params:
    data = input data array of integers
    num_elements = number of elements in the input array

- Return:
    Void, however upon return the array 'data' will hold the
    result of const_data_start / const_data_div
    (note: const_data_start should be divisible by const_data_div)
*/
__global__ void const_test_gpu_const_div(int * const data, const int num_elements) {
    const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid < num_elements) {
        int d = const_data_start;
        int d2 = const_data_div;
        d /= d2;
        data[tid] = d;
    }
}

// End - Constant Memory Functions
// #########################################################################################



// host functions
void execute_host_functions() {

}

// Constant memory work flow
void execute_constant_memory_functions() {

    int *data = NULL;
    const int num_threads = 32;
    const int num_blocks = WORK_SIZE / num_threads;

    int idata[WORK_SIZE], odata[WORK_SIZE];
    for(int i = 0; i < WORK_SIZE; i++) {
        idata[i] = i;
    }

    // #########################################################################################
    // Addition Global -> Constant Memory Test
    printf("\nRunning constant memory addition...\n");
    cudaMalloc((void** ) &data, sizeof(int) * WORK_SIZE);
    cudaMemcpy(data, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice);
    const_test_gpu_const_add<<<num_blocks, num_threads>>>(data, WORK_SIZE);
    cudaThreadSynchronize();
    cudaGetLastError();

    cudaMemcpy(odata, data, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost);

    printf("Output should be (input + %d)\n", const_data_start);
    for(int i = 0; i < WORK_SIZE; i++) {
        printf("Input val: %u, Output val: %u\n", idata[i], odata[i]);
    }

    cudaFree((void* ) data);
    cudaDeviceReset();
    // End addition test
    // #########################################################################################
    

    // #########################################################################################
    // Subtraction Global -> Constant Memory Test
    printf("\nRunning constant memory subtraction...\n");
    cudaMalloc((void** ) &data, sizeof(int) * WORK_SIZE);
    cudaMemcpy(data, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice);
    const_test_gpu_const_sub<<<num_blocks, num_threads>>>(data, WORK_SIZE);
    cudaThreadSynchronize();
    cudaGetLastError();

    cudaMemcpy(odata, data, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost);

    printf("Output should be (%d - input)\n", const_data_start);
    for(int i = 0; i < WORK_SIZE; i++) {
        printf("Input val: %u, Output val: %u\n", idata[i], odata[i]);
    }

    cudaFree((void* ) data);
    cudaDeviceReset();
    // End subtraction test
    // #########################################################################################


    // #########################################################################################
    // Multiplication Global -> Constant Memory Test
    printf("\nRunning constant memory multiplication...\n");
    cudaMalloc((void** ) &data, sizeof(int) * WORK_SIZE);
    cudaMemcpy(data, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice);
    const_test_gpu_const_mult<<<num_blocks, num_threads>>>(data, WORK_SIZE);
    cudaThreadSynchronize();
    cudaGetLastError();

    cudaMemcpy(odata, data, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost);

    printf("Output should be (input * %d)\n", const_data_mult);
    for(int i = 0; i < WORK_SIZE; i++) {
        printf("Input val: %u, Output val: %u\n", idata[i], odata[i]);
    }

    cudaFree((void* ) data);
    cudaDeviceReset();
    // End multiplication test
    // #########################################################################################


    // #########################################################################################
    // Division Global -> Constant Memory Test
    printf("\nRunning constant memory division...\n");
    cudaMalloc((void** ) &data, sizeof(int) * WORK_SIZE);
    cudaMemcpy(data, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice);
    const_test_gpu_const_div<<<num_blocks, num_threads>>>(data, WORK_SIZE);
    cudaThreadSynchronize();
    cudaGetLastError();

    cudaMemcpy(odata, data, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost);

    printf("Output should be (%d / %d)\n", const_data_start, const_data_div);
    for(int i = 0; i < WORK_SIZE; i++) {
        printf("Iteration: %d, Output val: %d\n", i, odata[i]);
    }

    cudaFree((void* ) data);
    cudaDeviceReset();
    // End subtraction test
    // #########################################################################################

    

}


// Shared memory work flow
void execute_shared_memory_functions() {

    const int len = ARR_LEN;
    int a[len], b[len], d[len];

    for (int i=0; i < len; i++) {
        a[i] = i * 2;
        b[i] = i;
        d[i] = 0;
    }

    // #########################################################################################
    // Addition Global -> Shared Memory Test
    int *d_d;
    int *d_d2;

    cudaMalloc(&d_d, len * sizeof(int));
    cudaMalloc(&d_d2, len * sizeof(int));

    printf("\nStarting shared memory addition test...\n\n");
    // Printing values
    for (int i=0; i < len; i++) {
        printf("Initial values: i=%d (a=%d, b=%d, d=%d)\n", i, a[i], b[i], d[i]);
    }

    printf("\nRunning shared memory addition...\n\n");
    cudaMemcpy(d_d, a, len*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d2, b, len*sizeof(int), cudaMemcpyHostToDevice);
    staticSharedAdd<<<1,len>>>(d_d, d_d2, len);
    cudaMemcpy(d, d_d, len*sizeof(int), cudaMemcpyDeviceToHost);

    // Output Values
    printf("c should equal a + b\n");
    for (int i=0; i < len; i++) {

        printf("Added values: i=%d (a=%d, b=%d, c=%d)\n", i, a[i], b[i], d[i]);
    }

    cudaFree((void* ) d_d);
    cudaFree((void* ) d_d2);
    cudaDeviceReset();
    // End of Addition Global -> Shared Memory Test
    // #########################################################################################

    // #########################################################################################
    // Multiplication Shared Memory Test
    int *md_d;

    cudaMalloc(&md_d, len * sizeof(int));

    printf("\nStarting shared memory multiplication test...\n\n");
    // Printing values
    for (int i=0; i < len; i++) {
        printf("Initial values: i=%d (a=%d, d=%d)\n", i, a[i], 0);
    }

    printf("\nRunning shared memory multiplication...\n");
    cudaMemcpy(md_d, a, len*sizeof(int), cudaMemcpyHostToDevice);
    staticSharedMult<<<1,len>>>(md_d, len);
    cudaMemcpy(d, md_d, len*sizeof(int), cudaMemcpyDeviceToHost);

    // Output Values
    printf("--- d should equal a * 2 ---\n");
    for (int i=0; i < len; i++) {
        printf("Multiplied Values: i=%d (a=%d, d=%d)\n", i, a[i], d[i]);
    }

    cudaFree((void* ) md_d);
    cudaDeviceReset();
    // End of Multiplication Global -> Shared Memory Test
    // #########################################################################################

    // #########################################################################################
    // Subraction Global -> Shared Memory Test
    int *sd_d;
    int *sd_d2;

    cudaMalloc(&sd_d, len * sizeof(int));
    cudaMalloc(&sd_d2, len * sizeof(int));

    printf("\nStarting shared memory subtraction test...\n\n");
    // Printing values
    for (int i=0; i < len; i++) {
        printf("Initial values: i=%d (a=%d, b=%d, d=%d)\n", i, a[i], b[i], 0);
    }

    printf("\nRunning shared memory addition...\n\n");
    cudaMemcpy(sd_d, a, len*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(sd_d2, b, len*sizeof(int), cudaMemcpyHostToDevice);
    staticSharedSub<<<1,len>>>(sd_d, sd_d2, len);
    cudaMemcpy(d, sd_d, len*sizeof(int), cudaMemcpyDeviceToHost);

    // Output Values
    printf("--- c should equal (a - b) ---\n");
    for (int i=0; i < len; i++) {
        printf("Subtracted values: i=%d (a=%d, b=%d, c=%d)\n", i, a[i], b[i], d[i]);
    }

    cudaFree((void* ) sd_d);
    cudaFree((void* ) sd_d2);
    cudaDeviceReset();
    // End of Subtraction Global -> Shared Memory Test
    // #########################################################################################

    // #########################################################################################
    // Division Global -> Shared Memory Test
    int *dd_d;
    int *dd_d2;

    cudaMalloc(&dd_d, len * sizeof(int));
    cudaMalloc(&dd_d2, len * sizeof(int));

    printf("\n Starting shared memory division test...\n\n");
    // Printing values
    for (int i=0; i < len; i++) {
        printf("Initial values: i=%d (a=%d, b=%d, d=%d)\n", i, a[i], b[i], 0);
    }

    printf("\nRunning shared memory division...\n\n");
    cudaMemcpy(dd_d, a, len*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dd_d2, b, len*sizeof(int), cudaMemcpyHostToDevice);
    staticSharedDiv<<<1,len>>>(sd_d, sd_d2, len);
    cudaMemcpy(d, sd_d, len*sizeof(int), cudaMemcpyDeviceToHost);

    // Output Values
    printf("--- c should equal (a / b) ---\n");
    for (int i=1; i < len; i++) {
        printf("Divided values: i=%d (a=%d, b=%d, c=%d)\n", i, a[i], b[i], d[i]);
    }

    cudaFree((void* ) dd_d);
    cudaFree((void* ) dd_d2);
    cudaDeviceReset();
    // End of Division Global -> Shared Memory Test
    // #########################################################################################

}

int main(void) {
    execute_host_functions();

    char ans;

    printf("test constant, test shared, or test both? (c/s/b): ");
    scanf("%c", &ans);

    if(ans == 's') {
        execute_shared_memory_functions();
    } else if (ans == 'c') {
        execute_constant_memory_functions();
    } else {
        execute_shared_memory_functions();
        execute_constant_memory_functions();
    }

    return 0;
}