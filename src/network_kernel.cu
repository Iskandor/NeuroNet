#include "network_kernel.cuh"
#include "Define.h"
#include <math.h>

__global__ void integrateKernel(double *p_output, const double *p_input, double *p_weights, int p_input_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    p_output[index] = 0;

	for(int i = 0; i < p_input_dim; i++) {
		p_output[index] += p_input[i] * p_weights[index * p_input_dim + i];
	}
}

/* vector addition */
__global__ void addKernel(double *ap, const double *p_output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    ap[index] += p_output[index];
}

/* activation function */
__global__ void activateKernel(double *p_output, double *p_input, const int *p_activationFunction) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    switch (*p_activationFunction) {
      case IDENTITY:
        p_output[index] = p_input[index];
        p_input[index] = 0;
      break;
      case BIAS:
        p_output[index] = -1;
        p_input[index] = 0;
      break;
      case BINARY:
        if (p_input[index] > 0) {
            p_output[index] = 1;
            p_input[index] = 0;
        }
        else {
            p_output[index] = 0;
        }
      break;
      case SIGMOID:
        p_output[index] = 1 / (1 + exp(-p_input[index]));
        p_input[index] = 0;
      break;
    }
}