#include "TDLambda.h"
#include "vector_kernel.cuh"
#include "Define.h"

TDLambda::TDLambda(NeuralNetwork* p_network, double p_lambda, double p_gamma) : BackProp(p_network) {
  _lambda = p_lambda;
  _gamma = p_gamma;

  for(unsigned int i = 0; i < _network->getConnections()->size(); i++) {
    int id = _network->getConnections()->at(i)->getId();
    int nCols = _network->getConnections()->at(i)->getInGroup()->getDim();
    int nRows = _network->getConnections()->at(i)->getOutGroup()->getDim();
    _delta[id] = new matrix2<double>(nRows, nCols);
    _delta[id]->set(0);
  }

  _Pt0 = new vectorN<double>(_network->getOutputGroup()->getDim());
  _Pt1 = new vectorN<double>(_network->getOutputGroup()->getDim());
  _Pt0->set(0);
  _Pt1->set(0);

  _input0 = new double[_network->getInputGroup()->getDim()];
  _input1 = nullptr;
}

TDLambda::~TDLambda() {
  for(unsigned int i = 0; i < _network->getConnections()->size(); i++) {
    int id = _network->getConnections()->at(i)->getId();
    delete _delta[id];
  }

  delete _Pt0;
  delete _Pt1;
}

double TDLambda::train(double* p_input, double* p_target) {
  double td_error = 0;

  if (_input1 != nullptr) {
    memcpy(_input0, _input1, sizeof(double) * _network->getInputGroup()->getDim());
  }
  _input1 = p_input;
    
  // forward activation phase
  _network->setInput(_input1);
  _network->onLoop();

  _Pt0->setVector(_Pt1->getVector());
  _Pt1->setVector(_network->getOutput());

  // calc TDerror
  for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
    td_error += p_target[i] + _gamma * _Pt1->at(i) - _Pt0->at(i);
  }

  // calc TD
  for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
    _prevGradient[_network->getOutputGroup()->getId()][i] = p_target[i] + _gamma * _Pt1->at(i) - _Pt0->at(i);
  }

  // updating phase for V(s)
  _network->setInput(_input0);
  _network->onLoop();
  backProp();

  return td_error;
}

double TDLambda::getTDerror(double* p_input, double* p_target) const {
  double td_error = 0;

  _network->setInput(p_input);
  _network->onLoop();

  for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
    td_error += p_target[i] + _gamma * _network->getOutput()[i] - _Pt0->at(i);
  }

  return td_error;
}

void TDLambda::updateWeights(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, Connection* p_connection) {

  cudaError_t cudaStatus;
  int nCols = p_inGroup->getDim();
  int nRows = p_outGroup->getDim();

  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks((int)ceil((double)nCols / threadsPerBlock.x), (int)ceil((double)nRows / threadsPerBlock.y));

  matrix2<double> delta(nRows, nCols);

  for(int i = 0; i < nRows; i++) {
    for(int j  = 0; j < nCols; j++) {
      delta.set(i, j, _gradient[p_outGroup->getId()][i] * p_inGroup->getOutput()[j]);
    }
  }

  double *dev_weights = 0;
  double *dev_et0 = 0;
  double *dev_et1 = 0;

  cudaStatus = cudaMalloc((void**)&dev_weights, nRows * nCols * sizeof(double));
  cudaStatus = cudaMalloc((void**)&dev_et0, nRows * nCols * sizeof(double));
  cudaStatus = cudaMalloc((void**)&dev_et1, nRows * nCols * sizeof(double));
  cudaStatus = cudaMemcpy(dev_weights, p_connection->getWeights()->getMatrix(), nRows * nCols * sizeof(double), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(dev_et0, _delta[p_connection->getId()]->getMatrix(), nRows * nCols * sizeof(double), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(dev_et1, delta.getMatrix(), nRows * nCols * sizeof(double), cudaMemcpyHostToDevice);

  mulConstVectorKernel<<<numBlocks, threadsPerBlock>>>(dev_et0, _lambda);
  cudaStatus = cudaGetLastError();
  cudaStatus = cudaDeviceSynchronize(); 

  addVectorKernel<<<numBlocks, threadsPerBlock>>>(dev_et1, dev_et0);
  cudaStatus = cudaGetLastError();
  cudaStatus = cudaDeviceSynchronize(); 

  mulConstVectorKernel<<<numBlocks, threadsPerBlock>>>(dev_et1, _alpha);
  cudaStatus = cudaGetLastError();
  cudaStatus = cudaDeviceSynchronize();  

  addVectorKernel<<<numBlocks, threadsPerBlock>>>(dev_weights, dev_et1);
  cudaStatus = cudaGetLastError();
  cudaStatus = cudaDeviceSynchronize();

  cudaStatus = cudaMemcpy(p_connection->getWeights()->getMatrix(), dev_weights, nRows * nCols * sizeof(double), cudaMemcpyDeviceToHost);
  cudaStatus = cudaMemcpy(_delta[p_connection->getId()]->getMatrix(), dev_et1, nRows * nCols * sizeof(double), cudaMemcpyDeviceToHost);

  cudaStatus = cudaFree(dev_weights);
  cudaStatus = cudaFree(dev_et0);
  cudaStatus = cudaFree(dev_et1);
}

