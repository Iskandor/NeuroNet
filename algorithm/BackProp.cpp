#include <math.h>
#include "../network/NeuralNetwork.h"
#include "BackProp.h"


BackProp::BackProp(NeuralNetwork* p_network) : GradientBase(p_network) {
  _alpha = 0;
  _weightDecay = 0;
  _momentum = 0;
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
    calcGradient(&_error);
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

  delta = _alpha * _delta[p_connection->getOutGroup()->getId()] * p_connection->getInGroup()->getOutput()->transpose();

  (*p_connection->getWeights()) += delta;
}

void BackProp::weightDecay(Connection* p_connection) const
{
  *p_connection->getWeights() *= (1 - _weightDecay);
}

void BackProp::setAlpha(double p_alpha) {
  _alpha = p_alpha;
}

void BackProp::setWeightDecay(double p_weightDecay) {
  _weightDecay = p_weightDecay;
}

void BackProp::setMomentum(double p_momentum)
{
  _momentum = p_momentum;
}
