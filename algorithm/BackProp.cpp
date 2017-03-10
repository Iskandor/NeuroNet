#include <math.h>
#include "../network/NeuralNetwork.h"
#include "BackProp.h"

using namespace NeuroNet;

BackProp::BackProp(NeuralNetwork* p_network, double p_weightDecay, double p_momentum, bool p_nesterov, const GRADIENT &p_gradient) : Optimizer(p_network, p_gradient, p_weightDecay, p_momentum, p_nesterov) {
}

BackProp::~BackProp(void) {
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
    calcGradient(&_error);

    for(auto it = _groupTree.rbegin(); it != _groupTree.rend(); ++it) {
        update(*it);
    }

    if (_batchSize > 1) {
        updateBatch();
    }

    return mse;
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
