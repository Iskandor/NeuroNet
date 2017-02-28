#include <math.h>
#include "../network/NeuralNetwork.h"
#include "BackProp.h"

using namespace NeuroNet;

BackProp::BackProp(NeuralNetwork* p_network, double p_weightDecay, double p_momentum, bool p_nesterov) : StochasticGradientDescent(p_network, p_momentum, p_nesterov), LearningAlgorithm() {
  _alpha = 0;
  _weightDecay = p_weightDecay;
  _input = nullptr;
  _error.resize(p_network->getOutputGroup()->getDim());
}

BackProp::~BackProp(void)
{
}

double BackProp::train(double *p_input, double* p_target) {
    double mse = 0;

    _input = p_input;
    
    // forward activation phase
    _network->setInput(p_input);
    _network->onLoop();

    // calc MSE
    for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
      mse += pow(p_target[i] - (*_network->getOutput())[i], 2);
    }

    // backward training phase
    for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
      _error[i] = p_target[i] - (*_network->getOutput())[i];
    }
    backProp();

    return mse;
}

void BackProp::backProp() {
    calcRegGradient(&_error);
    //calcNatGradient(0.001, &_error);
    for(auto it = _groupTree.rbegin(); it != _groupTree.rend(); ++it) {
      update(*it);
    }
}


void BackProp::update(NeuralGroup* p_node) {
    for(vector<int>::iterator it = p_node->getInConnections()->begin(); it != p_node->getInConnections()->end(); it++) {         
      updateWeights(_network->getConnections()->at(*it));
      if (_weightDecay != 0) weightDecay(_network->getConnections()->at(*it));
    }
}


void BackProp::updateWeights(Connection* p_connection) {
  int nCols = p_connection->getInGroup()->getDim();
  int nRows = p_connection->getOutGroup()->getDim();
  MatrixXd delta(nRows, nCols);

  delta = _alpha * _regGradient[p_connection->getId()];

  (*p_connection->getWeights()) += delta;
  (*p_connection->getOutGroup()->getBias()) += _delta[p_connection->getOutGroup()->getId()];
}

void BackProp::weightDecay(Connection* p_connection) const
{
  *p_connection->getWeights() *= (1 - _weightDecay);
}
