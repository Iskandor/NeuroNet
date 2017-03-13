#include <math.h>
#include "../network/NeuralNetwork.h"
#include "BackProp.h"

using namespace NeuroNet;

BackProp::BackProp(NeuralNetwork* p_network, double p_weightDecay, double p_momentum, bool p_nesterov, const GRADIENT &p_gradient) : Optimizer(p_network, p_gradient, p_weightDecay) {
    _momentum = p_momentum;
    _nesterov = p_nesterov;

    int nRows;
    int nCols;

    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); ++it) {
        nRows = it->second->getOutGroup()->getDim();
        nCols = it->second->getInGroup()->getDim();
        _v[it->second->getId()] = MatrixXd::Zero(nRows, nCols);
    }
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
    /*
    # Momentum update
    v = mu * v - learning_rate * dx # integrate velocity
    x += v # integrate position
    */

    MatrixXd v_prev;

    int id = p_connection->getId();

    if (_nesterov) {
        v_prev = _v[id].replicate(1,1);
    }

    _v[id] = _momentum * _v[id] + _alpha * (*_gradient)[id];

    /*
    v_prev = v # back this up
    v = mu * v - learning_rate * dx # velocity update stays the same
    x += -mu * v_prev + (1 + mu) * v # position update changes form
    */

    if (_nesterov) {
        (*p_connection->getWeights()) += -_momentum * v_prev + (1 + _momentum) * _v[id];
    }
    else {
        (*p_connection->getWeights()) += _v[id];
    }

    (*p_connection->getOutGroup()->getBias()) += _alpha * _delta[p_connection->getOutGroup()->getId()];

}
