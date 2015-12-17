#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void addVectorKernel(double *p_v1, const double *p_v2);
__global__ void subVectorKernel(double *p_v1, const double *p_v2);
__global__ void mulConstVectorKernel(double *p_vector, const double p_const);