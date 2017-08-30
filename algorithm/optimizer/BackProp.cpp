#include <math.h>
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
        _v[it->second->getId()] = Matrix::Zero(nRows, nCols);
    }
}

BackProp::~BackProp(void) {
}

double BackProp::train(Vector *p_input, Vector* p_target) {
    double mse = 0;
    
    // forward activation phase
    _network->activate(p_input);

    // backward training phase
    _error = (*p_target) - (*_network->getOutput());

    mse = calcMse(p_target);
    calcGradient(&_error);

    for(auto it = _groupTree.rbegin(); it != _groupTree.rend(); ++it) {
        update(*it);
    }

    return mse;
}

void BackProp::updateWeights(Connection* p_connection) {
    /*
    # Momentum update
    v = mu * v - learning_rate * dx # integrate velocity
    x += v # integrate position
    */
    int id = p_connection->getId();

    Matrix v_prev;

    if (_nesterov) {
        v_prev = Matrix(_v[id]);
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
