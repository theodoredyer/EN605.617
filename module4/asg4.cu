/*
Problem Spec: 
- Create a program that utilizes both host and global memory (both basic pageable and pinned memory)
- Implement a separate function that works with either type of global memory, which implements a simple Caesar cypher. 
- The array placed in both types of memory should include a number of characters.
- The function will use a defined offset (either hard coded or command line) and offset each character in the array by that value. 
- To test the validity of your function you should apply the additive inverse to get the original array back. 
*/

/* Theodore Dyer
Introduction to GPU Programming Spring 2022 (EN605.617.81)
Chance Pascale
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static const int WORK_SIZE = 12;

#define NUM_ELEMENTS 12
#define CAESAR_OFFSET 2

typedef struct {
    unsigned int a;
    unsigned int b;
    unsigned int c;
    unsigned int d;
} INTERLEAVED_T;

typedef INTERLEAVED_T INTERLEAVED_ARRAY_T[NUM_ELEMENTS];

typedef unsigned int ARRAY_MEMBER_T[NUM_ELEMENTS];

typedef struct {
	ARRAY_MEMBER_T a;
	ARRAY_MEMBER_T b;
	ARRAY_MEMBER_T c;
	ARRAY_MEMBER_T d;
} NON_INTERLEAVED_T;

__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

__host__ float add_test_interleaved_cpu(INTERLEAVED_T * const host_dest_ptr,
		const INTERLEAVED_T * const host_src_ptr, const unsigned int iter,
		const unsigned int num_elements) {
	cudaEvent_t start_time = get_time();
	for (unsigned int tid = 0; tid < num_elements; tid++) {
		printf("tid: %u ", tid);
		for (unsigned int i = 0; i < iter; i++) {
			host_dest_ptr[tid].a += host_src_ptr[tid].a;
			host_dest_ptr[tid].b += host_src_ptr[tid].b;
			host_dest_ptr[tid].c += host_src_ptr[tid].c;
			host_dest_ptr[tid].d += host_src_ptr[tid].d;
		}
	}

	cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);

	float delta = 0;
	cudaEventElapsedTime(&delta, start_time, end_time);

	return delta;
}

__global__
void applyCaesar(int n, int offset, float *x, float *y) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) y[i] = (char) ((int) y[i] + offset);
}

// Following bitreverse global_memory.cu model. 
__host__ __device__ char cyphershift(char letter) {
    letter = ((int) letter);
    letter = letter + CAESAR_OFFSET;
    return ((char) letter);
}

__global__ void cyphershift(void *data) {
    char *idata = (char*) data;
    idata[threadIdx.x] = cyphershift(idata[threadIdx.x]);
}

void execute_host_functions() { 
    INTERLEAVED_T host_dest_ptr[NUM_ELEMENTS];
    INTERLEAVED_T host_src_ptr[NUM_ELEMENTS];
    float call_time = add_test_interleaved_cpu(host_dest_ptr, host_src_ptr, 4, NUM_ELEMENTS);
    printf("elapsed: %fms\n", call_time);
}

void execute_gpu_functions() {
    cudaEvent_t start_time = get_time();
    void *d = NULL;
    char idata[WORK_SIZE], odata[WORK_SIZE];
    int i;
    for(i=0; i < WORK_SIZE; i++){
        idata[i] = ((char) i + 97);
        printf("char: %c\n", ((char) i + 97));
    }

    cudaMalloc((void**) &d, sizeof(char) * WORK_SIZE);
    cudaMemcpy(d, idata, sizeof(char) * WORK_SIZE, cudaMemcpyHostToDevice);
    cyphershift<<<1, WORK_SIZE, WORK_SIZE * sizeof(char)>>>(d);

    cudaThreadSynchronize();
    cudaGetLastError();

    cudaMemcpy(odata, d, sizeof(char) * WORK_SIZE, cudaMemcpyDeviceToHost);

    for(i = 0; i < WORK_SIZE; i++) {
        printf("Input Value: %c, device output: %c, host output: %c\n", idata[i], odata[i], cyphershift(idata[i]));
    }

    for (i=0; i < WORK_SIZE; i++){
        if(i==0) printf("\nInput String: ");
        printf("%c", idata[i]);
        if(i==WORK_SIZE - 1) printf("\n");
    }
    for (i=0; i < WORK_SIZE; i++){
        if(i==0) printf("Output String: ");
        printf("%c", odata[i]);
        if(i==WORK_SIZE - 1) printf("\n");
    }
    for (i=0; i < WORK_SIZE; i++){
        if(i==0) printf("Additive Inverse Output String: ");
        printf("%c", (char) ((int) odata[i] - CAESAR_OFFSET));
        if(i==WORK_SIZE - 1) printf("\n\n");
    }

    
    

    cudaEvent_t end_time = get_time();
	cudaEventSynchronize(end_time);

	float delta = 0;
	float timepassed = cudaEventElapsedTime(&delta, start_time, end_time);
    printf("elapsed: %fms\n", timepassed);

    cudaFree((void* )d);
    cudaDeviceReset();
}

int main(void) {

    execute_host_functions();
    execute_gpu_functions();
    return 0;
}