#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void integrateKernel(double *p_output, const double *p_input, double *p_weights, int p_input_dim);
__global__ void addKernel(double *ap, const double *p_output);
__global__ void activateKernel(double *p_output, double *p_input, const int *p_activationFunction);