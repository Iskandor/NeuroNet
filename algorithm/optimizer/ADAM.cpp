//
// Created by mpechac on 13. 3. 2017.
//

#include "ADAM.h"

using namespace NeuroNet;


ADAM::ADAM(NeuralNetwork *p_network, double p_beta1, double p_beta2, double p_epsilon, const GradientDescent::GRADIENT &p_gradient) : Optimizer(p_network, p_gradient) {
    _beta1 = p_beta1;
    _beta2 = p_beta2;

    int nRows;
    int nCols;

    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); ++it) {
        nRows = it->second->getOutGroup()->getDim();
        nCols = it->second->getInGroup()->getDim();
        _m[it->second->getId()] = Matrix::Zero(nRows, nCols);
        _v[it->second->getId()] = Matrix::Zero(nRows, nCols);
        _eps[it->second->getId()] = Matrix::Value(nRows, nCols, p_epsilon);
        _mCorr[it->second->getId()] = Matrix::Value(nRows, nCols, 1 / (1 - _beta1));
        _vCorr[it->second->getId()] = Matrix::Value(nRows, nCols, 1 / (1 - _beta2));
    }
}

ADAM::~ADAM() {

}

double ADAM::train(Vector *p_input, Vector *p_target) {
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

    return mse;
}

void ADAM::updateWeights(Connection *p_connection) {
    /*
    m = beta1*m + (1-beta1)*dx
    v = beta2*v + (1-beta2)*(dx**2)
    x += - learning_rate * m / (np.sqrt(v) + eps)
    */

    int id = p_connection->getId();
    Matrix gpow2 = (*_gradient)[id].ew_pow(2);

    _m[id] = _beta1 * _m[id] + (1 - _beta1) * (*_gradient)[id];
    _v[id] = _beta2 * _v[id] + (1 - _beta2) * gpow2;

    Matrix vsqrt = _v[id].ew_dot(_vCorr[id]).ew_sqrt();
    Matrix gdiv = (vsqrt + _eps[id]).inv();

    (*p_connection->getWeights()) += _alpha * _m[id].ew_dot(_mCorr[id]).ew_dot(gdiv);
    (*p_connection->getOutGroup()->getBias()) += _alpha * _delta[p_connection->getOutGroup()->getId()];
}
