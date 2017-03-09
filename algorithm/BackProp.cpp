#include <math.h>
#include "../network/NeuralNetwork.h"
#include "BackProp.h"

using namespace NeuroNet;

BackProp::BackProp(NeuralNetwork* p_network, double p_weightDecay, double p_momentum, bool p_nesterov) : GradientDescent(p_network, p_momentum, p_nesterov) {
  _alpha = 0;
  _weightDecay = p_weightDecay;
  _error = VectorXd::Zero(p_network->getOutput()->size());
}

BackProp::~BackProp(void)
{
}

double BackProp::train(VectorXd *p_input, VectorXd* p_target) {
    double mse = 0;
    
    // forward activation phase
    _network->activate(p_input);

    // backward training phase
    for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
      _error[i] = (*p_target)[i] - (*_network->getOutput())[i];
    }

    mse = calcMse(p_target);

    calcRegGradient(&_error);
    //calcNatGradient(0.001, &_error);
    for(auto it = _groupTree.rbegin(); it != _groupTree.rend(); ++it) {
        update(*it);
    }

    updateBatch();

    return mse;
}

void BackProp::update(NeuralGroup* p_node) {
    for(vector<int>::iterator it = p_node->getInConnections()->begin(); it != p_node->getInConnections()->end(); it++) {
      updateWeights(_network->getConnections()->at(*it));
      if (_weightDecay != 0) weightDecay(_network->getConnections()->at(*it));
    }
}

void BackProp::updateWeights(Connection* p_connection) {
    if (_batchSize == 1) {
        (*p_connection->getWeights()) += _alpha * _regGradient[p_connection->getId()];
        (*p_connection->getOutGroup()->getBias()) += _alpha * _delta[p_connection->getOutGroup()->getId()];
    }
    else {
        if (_batch < _batchSize) {
            _weightDelta[p_connection->getId()] += _alpha * _regGradient[p_connection->getId()];
            _biasDelta[p_connection->getId()] += _delta[p_connection->getOutGroup()->getId()];
        }
        else {
            (*p_connection->getWeights()) += _weightDelta[p_connection->getId()];
            (*p_connection->getOutGroup()->getBias()) += _biasDelta[p_connection->getId()];

            _weightDelta[p_connection->getId()].fill(0);
            _biasDelta[p_connection->getId()].fill(0);
        }

    }
}

void BackProp::weightDecay(Connection* p_connection) const
{
  *p_connection->getWeights() *= (1 - _weightDecay);
}

double BackProp::calcMse(VectorXd *p_target) {
    double mse = 0;
    // calc MSE
    for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
        mse += pow((*p_target)[i] - (*_network->getOutput())[i], 2);
    }

    return mse;
}

void BackProp::updateBatch() {
    if (_batch < _batchSize) {
        _batch++;
    }
    else {
        _batch = 0;
    }
}
