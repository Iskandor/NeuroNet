#include "BackProp.h"
#include <math.h>

BackProp::BackProp(NeuralNetwork* p_network)
{
  _network = p_network;

  for(unsigned int i = 0; i < p_network->getGroups()->size(); i++) {
    int id = p_network->getGroups()->at(i)->getId();
    _error[id] = new double[p_network->getGroups()->at(i)->getDim()];
    _deriv[id] = new double[p_network->getGroups()->at(i)->getDim()];
    _prevError[id] = new double[p_network->getGroups()->at(i)->getDim()];
  }
}


BackProp::~BackProp(void)
{
  for(unsigned int i = 0; i < _network->getGroups()->size(); i++) {
    int id = _network->getGroups()->at(i)->getId();
    delete[] _error[id];
    delete[] _deriv[id];
    delete[] _prevError[id];
  }
}

double BackProp::train(double *p_input, double* p_target) {
    double mse = 0;

    _input = p_input;
    
    // forward activation phase
    _network->setInput(p_input);
    _network->onLoop();

    // calc MSE
    for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
      mse += pow(p_target[i] - _network->getOutput()[i], 2);
    }

    // backward training phase
    for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
      _prevError[_network->getOutputGroup()->getId()][i] = p_target[i] - _network->getOutput()[i];
    }
    backProp();

    return mse;
}

void BackProp::backProp() {
    /* invalidate all neural groups */
    for(vector<NeuralGroup*>::iterator it = _network->getGroups()->begin(); it != _network->getGroups()->end(); it++) {
        ((NeuralGroup*)(*it))->invalidate();
    }
    backActivate(_network->getOutputGroup());
}

void BackProp::backActivate(NeuralGroup* p_node) {
    calcDeriv(p_node);
    calcError(p_node);
    
    /* send error signal to synapsis and repeat it for not activated group to prevent infinite loops */
    for(vector<int>::iterator it = p_node->getInConnections()->begin(); it != p_node->getInConnections()->end(); it++) {
        if (!_network->getConnections()->at(*it)->getInGroup()->isActivated()) {            
            updateWeights(_network->getConnections()->at(*it)->getInGroup(), p_node, _network->getConnections()->at(*it));            
            calcPrevError(_network->getConnections()->at(*it)->getInGroup(), p_node, _network->getConnections()->at(*it));
            backActivate(_network->getConnections()->at(*it)->getInGroup());
        }
    }
}

void BackProp::updateWeights(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, Connection* p_connection) {
  
  int nCols = p_inGroup->getDim();
  int nRows = p_outGroup->getDim();

  dim3 dimBlock(nRows, nCols);
  dim3 dimGrid((int)ceil(nRows/dimBlock.x),(int)ceil(p_outGroup->getDim()/dimBlock.y));

  matrix<double> delta(nRows, nCols);

  for(int i = 0; i < nRows; i++) {
    for(int j  = 0; j < nCols; j++) {
      delta.set(i, j, _alpha * _error[p_outGroup->getId()][i] * p_inGroup->getOutput()[j]);
    }
  }
  
  //p_connection->getWeights()->set(j,i, p_connection->getWeights()->at(j,i) + );
  int    *dev_colDim = 0;
  double *dev_weights = 0;
  double *dev_delta = 0;
  double *dev_output = 0;

  cudaStatus = cudaMalloc((void**)&dev_colDim, sizeof(int));
  cudaStatus = cudaMalloc((void**)&dev_output, nRows * nCols * sizeof(double));
  cudaStatus = cudaMalloc((void**)&dev_weights, nRows * nCols * sizeof(double));
  cudaStatus = cudaMalloc((void**)&dev_delta, nRows * nCols * sizeof(double));

  cudaStatus = cudaMemcpy(dev_weights, p_connection->getWeights()->getMatrix(), nRows * nCols * sizeof(double), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(dev_colDim, &nCols, sizeof(int), cudaMemcpyHostToDevice);
  activateKernel<<<dimGrid,dimBlock>>>(dev_output, dev_weights, dev_delta, dev_colDim);
  cudaStatus = cudaGetLastError();
  cudaStatus = cudaDeviceSynchronize();  

  cudaStatus = cudaMemcpy(p_connection->getWeights()->getMatrix(), dev_output, nRows * nCols * sizeof(double), cudaMemcpyDeviceToHost);

  cudaStatus = cudaFree(dev_colDim);
  cudaStatus = cudaFree(dev_weights);
  cudaStatus = cudaFree(dev_delta);
  cudaStatus = cudaFree(dev_output);
}

void BackProp::calcError(NeuralGroup* p_group) {
  int id = p_group->getId();  
  
  for(int i = 0; i < p_group->getDim(); i++) {
    _error[id][i] = _deriv[id][i] * _prevError[id][i];
  }
}

void BackProp::calcDeriv(NeuralGroup* p_group) {
  for(int i = 0; i < p_group->getDim(); i++) {
    _deriv[p_group->getId()][i] = p_group->getOutput()[i] * (1 - p_group->getOutput()[i]);
  }
}

void BackProp::calcPrevError(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, Connection* p_connection) {
  int inId = p_inGroup->getId();
  int outId = p_outGroup->getId();

  for(int i = 0; i < p_inGroup->getDim(); i++) {
    _prevError[inId][i] = 0;
    for(int j = 0; j < p_outGroup->getDim(); j++) {
      _prevError[inId][i] += _error[outId][j] * p_connection->getWeights()->at(j, i);
    }
  }
}

void BackProp::setAlpha(double p_alpha) {
  _alpha = p_alpha;
}
