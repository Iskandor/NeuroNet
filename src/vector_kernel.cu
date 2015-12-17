#include "vector_kernel.cuh"

/* vector addition */
__global__ void addVectorKernel(double *p_v1, const double *p_v2) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    p_v1[index] += p_v2[index];
}

/* vector subtraction */
__global__ void subVectorKernel(double *p_v1, const double *p_v2) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    p_v1[index] -= p_v2[index];
}

/* multiply vector by constant */
__global__ void mulConstVectorKernel(double *p_vector, const double p_const) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    p_vector[index] *= p_const;  
}

