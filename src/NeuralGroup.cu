#include "NeuralGroup.cuh"
#include "network_kernel.cuh"
#include "vector_kernel.cuh"
#include "Define.h"

#include <memory>

using namespace std;

NeuralGroup::NeuralGroup(int p_id, int p_dim, int p_activationFunction)
{
  _id = p_id;
  _dim = p_dim;
  _activationFunction = p_activationFunction;

  _output = new double[p_dim];
  memset(_output, 0, sizeof(double)*p_dim);

  if (_activationFunction == BIAS) {
    for(int i = 0; i < p_dim; i++) {
      _output[i] = 1;
    }
  }

  _actionPotential = new double[p_dim];
  memset(_actionPotential, 0, sizeof(double)*p_dim);

  _valid = false;
}


NeuralGroup::~NeuralGroup(void)
{
	delete[] _actionPotential;
  delete[] _output;
}

void NeuralGroup::init() {
}

/* calculate output of group */
void NeuralGroup::fire() {
    _valid = true;
    activate(_actionPotential, _activationFunction);
}

void NeuralGroup::addInConnection(int p_index) {
    _inConnections.push_back(p_index);
}

void NeuralGroup::addOutConnection(int p_index) {
    _outConnections.push_back(p_index);
}

/* wrapper for CUDA function which will calculate the sum of inputs for each neuron */
cudaError_t NeuralGroup::integrate(double *p_input, double *p_weights, int p_input_dim) {
  cudaError_t cudaStatus;
  
  double *dev_input = 0;
  double *dev_weights = 0;
  double *dev_output = 0;
  double *dev_ac = 0;

  cudaStatus = cudaMalloc((void**)&dev_input, p_input_dim * sizeof(double));
  cudaStatus = cudaMalloc((void**)&dev_weights, _dim * p_input_dim * sizeof(double));
  cudaStatus = cudaMalloc((void**)&dev_output, _dim * sizeof(double));
  cudaStatus = cudaMalloc((void**)&dev_ac, _dim * sizeof(double));
  cudaStatus = cudaMemcpy(dev_ac, _actionPotential, _dim * sizeof(double), cudaMemcpyHostToDevice);

  cudaStatus = cudaMemcpy(dev_input, p_input, p_input_dim * sizeof(double), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(dev_weights, p_weights, _dim * p_input_dim * sizeof(double), cudaMemcpyHostToDevice);

  integrateKernel<<<(int)ceil((double)_dim/(double)MAX_THREAD),_dim>>>(dev_output, dev_input, dev_weights, p_input_dim);
  cudaStatus = cudaGetLastError();
  cudaStatus = cudaDeviceSynchronize();
    
  addVectorKernel<<<(int)ceil((double)_dim/(double)MAX_THREAD),_dim>>>(dev_ac, dev_output);
  cudaStatus = cudaGetLastError();
  cudaStatus = cudaDeviceSynchronize();

  cudaStatus = cudaMemcpy(_actionPotential, dev_ac, _dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaStatus = cudaFree(dev_input);
  cudaStatus = cudaFree(dev_weights);
  cudaStatus = cudaFree(dev_output);
  cudaStatus = cudaFree(dev_ac);

	return cudaStatus;
}

/* function which should calculate the output of neuron (activation function output) according to action potential */
cudaError_t NeuralGroup::activate(double* p_input, const int p_activationFunction) {
  cudaError_t cudaStatus;

  int    *dev_activationFunction = 0;
  double *dev_input = 0;
  double *dev_output = 0;
  cudaStatus = cudaMalloc((void**)&dev_activationFunction, sizeof(int));
  cudaStatus = cudaMalloc((void**)&dev_input, _dim * sizeof(double));
  cudaStatus = cudaMalloc((void**)&dev_output, _dim * sizeof(double));

  cudaStatus = cudaMemcpy(dev_input, p_input, _dim * sizeof(double), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(dev_activationFunction, &p_activationFunction, sizeof(int), cudaMemcpyHostToDevice);
  activateKernel<<<(int)ceil((double)_dim/(double)MAX_THREAD),_dim>>>(dev_output, dev_input, dev_activationFunction);
  cudaStatus = cudaGetLastError();
  cudaStatus = cudaDeviceSynchronize();
  cudaStatus = cudaMemcpy(_output, dev_output, _dim * sizeof(double), cudaMemcpyDeviceToHost);
  cudaStatus = cudaMemcpy(p_input, dev_input, _dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaStatus = cudaFree(dev_activationFunction);
  cudaStatus = cudaFree(dev_input);
  cudaStatus = cudaFree(dev_output);

  return cudaStatus;
}