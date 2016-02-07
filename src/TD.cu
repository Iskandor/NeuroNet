#include "Define.h"
#include "TD.h"
#include "vector_kernel.cuh"

TD::TD(NeuralNetwork* p_network)
{
  _network = p_network;
  _alpha = 0;
  _lambda = 0;
  _t = 0;

  for(unsigned int i = 0; i < p_network->getConnections()->size(); i++) {
    auto id = p_network->getConnections()->at(i)->getId();
    int nRows = p_network->getConnections()->at(i)->getOutGroup()->getDim();
    int nCols = p_network->getConnections()->at(i)->getInGroup()->getDim();

    _eligibility[id] = new matrix3<double>(_network->getOutputGroup()->getDim(), nRows, nCols);
    _eligibility[id]->set(0);
    _delta[id] = new matrix2<double>(nRows, nCols);
  }

  for(unsigned int i = 0; i < p_network->getGroups()->size(); i++) {
    auto id = p_network->getGroups()->at(i)->getId();
    int outputId = _network->getOutputGroup()->getId();
    int nRows = p_network->getGroups()->at(i)->getDim();

    _deriv[id] = new double[p_network->getGroups()->at(i)->getDim()];
    _eligDelta[outputId][id] = new matrix2<double>(_network->getOutputGroup()->getDim(), nRows);
    _eligDelta[outputId][id]->set(1);
  }

  _outputT = new double[p_network->getOutputGroup()->getDim()];
  memset(_outputT, 0, sizeof(double)*p_network->getOutputGroup()->getDim());

  _error = new double[p_network->getOutputGroup()->getDim()];
  memset(_error, 0, sizeof(double)*p_network->getOutputGroup()->getDim());
}

TD::~TD(void)
{
  auto outputId = _network->getOutputGroup()->getId();

  for(unsigned int i = 0; i < _network->getConnections()->size(); i++) {
    auto id = _network->getConnections()->at(i)->getId();
    delete _eligibility[id];
    delete _delta[id];
  }

  for(unsigned int i = 0; i < _network->getGroups()->size(); i++) {
    auto id = _network->getGroups()->at(i)->getId();
    delete[] _deriv[id];
    delete _eligDelta[outputId][id];
  }

  delete[] _outputT;
  delete[] _error;
}

void TD::train(double* p_input, double *p_output, double *p_nextOutput)
{
  _network->setInput(p_input);

  calcError(p_output, p_nextOutput);
  TDRec(_network->getOutputGroup());
}

void TD::TDRec(NeuralGroup* p_node)
{
  calcDeriv(p_node);
  for(auto i = 0; i < p_node->getInConnections()->size(); i++)
  {
    Connection* connection = _network->getConnections()->at(p_node->getInConnections()->at(i));
    calcEligDelta(connection);
    updateEligibility(connection);
    calcDelta(connection);
    updateWeights(connection->getInGroup(), connection->getOutGroup(), connection);
  }

  for(auto i = 0; i < p_node->getInConnections()->size(); i++)
  {
    Connection* connection = _network->getConnections()->at(p_node->getInConnections()->at(i));
    TDRec(connection->getInGroup());
  }  
}

void TD::updateWeights(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, Connection* p_connection) {
  cudaError_t cudaStatus;
  int nCols = p_inGroup->getDim();
  int nRows = p_outGroup->getDim();

  // update kernel
  double *dev_delta = nullptr;
  double *dev_weights = nullptr;

  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks(static_cast<int>(ceil(static_cast<double>(nCols) / threadsPerBlock.x)), static_cast<int>(ceil(static_cast<double>(nRows) / threadsPerBlock.y)));

  cudaStatus = cudaMalloc(reinterpret_cast<void**>(&dev_delta), nRows * nCols * sizeof(double));
  cudaStatus = cudaMalloc(reinterpret_cast<void**>(&dev_weights), nRows * nCols * sizeof(double));
  cudaStatus = cudaMemcpy(dev_delta, _delta[p_connection->getId()], nRows * nCols * sizeof(double), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(dev_weights, p_connection->getWeights()->getMatrix(), nRows * nCols * sizeof(double), cudaMemcpyHostToDevice);

  mulConstVectorKernel<<<numBlocks, threadsPerBlock>>>(dev_delta, _alpha);
  cudaStatus = cudaGetLastError();
  cudaStatus = cudaDeviceSynchronize();

  addVectorKernel<<<numBlocks, threadsPerBlock>>>(dev_weights, dev_delta);
  cudaStatus = cudaGetLastError();
  cudaStatus = cudaDeviceSynchronize();

  cudaStatus = cudaMemcpy(_delta[p_connection->getId()], dev_delta, nRows * nCols * sizeof(double), cudaMemcpyDeviceToHost);
  cudaStatus = cudaMemcpy(p_connection->getWeights()->getMatrix(), dev_weights, nRows * nCols * sizeof(double), cudaMemcpyDeviceToHost);

  cudaStatus = cudaFree(dev_delta);
  cudaStatus = cudaFree(dev_weights);
}

void TD:: calcError(double *p_output, double* p_nextOutput) const
{
  cudaError_t cudaStatus;
  
  int dim = _network->getOutputGroup()->getDim();
  double *dev_output = nullptr;
  double *dev_outputT = nullptr;

  cudaStatus = cudaMalloc(reinterpret_cast<void**>(&dev_output), dim * sizeof(double));
  cudaStatus = cudaMalloc(reinterpret_cast<void**>(&dev_outputT), dim * sizeof(double));
  cudaStatus = cudaMemcpy(dev_output, p_output, dim * sizeof(double), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(dev_outputT, p_nextOutput, dim * sizeof(double), cudaMemcpyHostToDevice);

  subVectorKernel<<<static_cast<int>(ceil(static_cast<double>(dim)/static_cast<double>(MAX_THREAD))),dim>>>(dev_output, dev_outputT);
  cudaStatus = cudaGetLastError();
  cudaStatus = cudaDeviceSynchronize();

  cudaStatus = cudaMemcpy(_error, dev_output, dim * sizeof(double), cudaMemcpyDeviceToHost);
  cudaStatus = cudaFree(dev_output);
  cudaStatus = cudaFree(dev_outputT);
}

void TD::calcDelta(Connection* p_connection)
{
  for(auto j = 0; j < p_connection->getOutGroup()->getDim(); j++)
  {
    for(auto k = 0; k < p_connection->getInGroup()->getDim(); k++)
    {
      double value = 0;
      for(auto i = 0; i < _network->getOutputGroup()->getDim(); i++)
      {  
        value += _error[i] * _eligibility[p_connection->getId()]->at(i, j, k);
      }
      _delta[p_connection->getId()]->set(j, k, value);
    }
  } 
}


void TD::updateEligibility(Connection* p_connection) {
  int outId = _network->getOutputGroup()->getId();

  for(auto i = 0; i < _network->getOutputGroup()->getDim(); i++)
  {    
    for(auto j = 0; j < p_connection->getOutGroup()->getDim(); j++)
    {
      for(auto k = 0; k < p_connection->getInGroup()->getDim(); k++)
      {
        double value = _lambda * _eligibility[p_connection->getId()]->at(i, j, k) + _eligDelta[outId][p_connection->getOutGroup()->getId()]->at(i, j) * p_connection->getInGroup()->getOutput()[k];
        _eligibility[p_connection->getId()]->set(i, j, k, value);
      }
    }
  }
}

void TD::calcEligDelta(Connection* p_connection) {
  int outId = _network->getOutputGroup()->getId();

  for(auto i = 0; i < _network->getOutputGroup()->getDim(); i++)
  {    
    for(auto j = 0; j < p_connection->getOutGroup()->getDim(); j++)
    {
      for(auto k = 0; k < p_connection->getInGroup()->getDim(); k++)
      {
        double value = _eligDelta[outId][p_connection->getOutGroup()->getId()]->at(i, j) * p_connection->getWeights()->at(j, k) * _deriv[p_connection->getOutGroup()->getId()][k];
        _eligDelta[outId][p_connection->getInGroup()->getId()]->set(i, k, value);
      }    
    }
  }
}

void TD::calcDeriv(NeuralGroup* p_group) {
  for(auto i = 0; i < p_group->getDim(); i++) {
    switch(p_group->getActivationFunction()) {
      case SIGMOID:
        _deriv[p_group->getId()][i] = p_group->getDim();
      break;
    }    
  }
}


void TD::setAlpha(double p_alpha) {
  _alpha = p_alpha;
}

void TD::setLambda(double p_lambda) {
  _lambda = p_lambda;
}
