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
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <curand.h>
#include <curand_kernel.h>

#define MAX 100
#define N 10
#define M 10



// Following example in curand_example.cu for setting up curand operations.

// Initializes random states
__global__ void init(unsigned int seed, curandState_t* states) {
    const int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    curand_init(seed, thread_idx, 0, &states[thread_idx]);
}

// Kernel that takes an array of states, array of ints, and puts a random int into each
__global__ void generate_random(curandState_t* states, int* numbers) {
    const int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    numbers[thread_idx] = curand(&states[thread_idx]) % MAX;
}


// ** Via cusolver_example.cu **
void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}


/*
execute_rand()
- Runs tests associated with the curand library

- Params:
    x_dim: number of rows for which to generate values
    y_dim: number of columns for which to generate values
*/
void execute_rand(int x_dim, int y_dim)
{
    
    printf("--------------------------------------------------\n");
    printf("\nRunning random number operation... \n\n");

    // Initialize time for tracking runtime
    cudaEvent_t start, stop;
    float elapsed_time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Initialize and allocate space for states
    curandState_t* states;
    cudaMalloc((void **) &states, x_dim * y_dim * sizeof(curandState_t));

    // Initialize random states via kernel
    init<<<x_dim,y_dim>>>(time(0), states);

    // Allocate an array of ints to be used with states to generate numbers
    int cpu_nums[x_dim*y_dim];
    int* gpu_nums;
    cudaMalloc((void **) &gpu_nums, x_dim * y_dim *sizeof(int));

    // Generate randoms via kernel
    generate_random<<<x_dim,y_dim>>>(states, gpu_nums);

    // Copy data back to host
    cudaMemcpy(cpu_nums, gpu_nums, x_dim * y_dim * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);


    /* print them out */
    printf("Printing out [ %d x %d ] random numbers: \n", x_dim, y_dim);
    for (int i = 0; i < x_dim; i++) {
        for (int j =0; j < y_dim; j++){
            printf("%u ", cpu_nums[i * x_dim + j]);
        }
        printf("\n");
    }

    printf("\nExecution time taken: %3.1f ms\n\n", elapsed_time);
    printf("--------------------------------------------------\n");

    cudaFree(states);
    cudaFree(gpu_nums);

}

void execute_solve()
{
	cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;    
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    const int dim = 3;
    const int lda = dim;
    const int ldb = dim;
    const int nrhs = 1;

    printf("--------------------------------------------------\n");
    printf("\nRunning matrix multiplication operation... \n\n");

    

    // The calculation we're seeing to find 
    /*
        Exe1:
            | 9 8 7 |
        A = | 6 5 4 |
            | 3 2 1 |
        
        x = [ 1 1 1 ]
        b = [ 24 15 6 ]
    */

    double A[lda*dim] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 3.0, 6.0, 1.0}; 
    double B[ldb*nrhs] = { 42.0, 105.0, 28.0}; 
    double XC[ldb*nrhs]; // solution matrix from GPU

    double *d_A = NULL; // linear memory of GPU  
    double *d_tau = NULL; // linear memory of GPU 
    double *d_B  = NULL; 
    int *devInfo = NULL; // info in gpu (device copy)
    double *d_work = NULL;
    int  lwork = 0; 

    int info_gpu = 0;

    const double one = 1;

    printf("Matrix A : \n");
    printMatrix(dim, dim, A, lda, "A");
    printf("\n");
    printf("Matrix B : \n");
    printMatrix(dim, nrhs, B, lda, "B");
    printf("\n");

    // Initialize time for tracking runtime
    cudaEvent_t start, stop;
    float elapsed_time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Following the guide in cusolver_example.cu

    // First, create handles
    // --------------------------------------------------------------------
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    // --------------------------------------------------------------------

    // Copy matrices to device memory
    // --------------------------------------------------------------------
    cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(double) * lda * dim);
    cudaStat2 = cudaMalloc ((void**)&d_tau, sizeof(double) * dim);
    cudaStat3 = cudaMalloc ((void**)&d_B  , sizeof(double) * ldb * nrhs);
    cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * dim   , cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(double) * ldb * nrhs, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    // --------------------------------------------------------------------

    // Query geqrf and ormqr
    // --------------------------------------------------------------------
    cusolver_status = cusolverDnDgeqrf_bufferSize(
        cusolverH, 
        dim, 
        dim, 
        d_A, 
        lda, 
        &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
 
    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);
    // --------------------------------------------------------------------

    // QR Factorization
    // --------------------------------------------------------------------
    cusolver_status = cusolverDnDgeqrf(
        cusolverH, 
        dim, 
        dim, 
        d_A, 
        lda, 
        d_tau, 
        d_work, 
        lwork, 
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    // check if QR is good or not
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(0 == info_gpu);
    // --------------------------------------------------------------------

    // Compute Q ^ (T * B)
    // --------------------------------------------------------------------
    cusolver_status= cusolverDnDormqr(
        cusolverH, 
        CUBLAS_SIDE_LEFT, 
        CUBLAS_OP_T,
        dim, 
        nrhs, 
        dim, 
        d_A, 
        lda,
        d_tau,
        d_B,
        ldb,
        d_work,
        lwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    assert(0 == info_gpu);
    // --------------------------------------------------------------------

    // Compute X = ( R \ Q ^ (T * B))
    // --------------------------------------------------------------------
    cublas_status = cublasDtrsm(
         cublasH,
         CUBLAS_SIDE_LEFT,
         CUBLAS_FILL_MODE_UPPER,
         CUBLAS_OP_N, 
         CUBLAS_DIAG_NON_UNIT,
         dim,
         nrhs,
         &one,
         d_A,
         lda,
         d_B,
         ldb);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(XC, d_B, sizeof(double)*ldb*nrhs, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("\nExecution time taken: %3.1f ms\n\n", elapsed_time);
    printf("--------------------------------------------------\n");

    printf("X = \n");
    printMatrix(dim, nrhs, XC, ldb, "X");
    printf("\n");
    // --------------------------------------------------------------------

}


int main(void) {

    // Fill in any row/col values for desired random matrix dimensions
	execute_rand(12,12);
    execute_rand(24,24);
    execute_solve();

	return EXIT_SUCCESS;
}