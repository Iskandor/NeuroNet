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
        _m[it->second->getId()] = MatrixXd::Zero(nRows, nCols);
        _v[it->second->getId()] = MatrixXd::Zero(nRows, nCols);
        _eps[it->second->getId()] = MatrixXd::Zero(nRows, nCols);
        _eps[it->second->getId()].setOnes();
        _eps[it->second->getId()] *= p_epsilon;
    }
}

ADAM::~ADAM() {

}

double ADAM::train(VectorXd *p_input, VectorXd *p_target) {
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

void ADAM::updateWeights(Connection *p_connection) {
    /*
    m = beta1*m + (1-beta1)*dx
    v = beta2*v + (1-beta2)*(dx**2)
    x += - learning_rate * m / (np.sqrt(v) + eps)
    */

    int id = p_connection->getId();
    MatrixXd gpow2 = (*_gradient)[id].cwiseProduct((*_gradient)[id]);

    _m[id] = _beta1 * _m[id] + (1 - _beta1) * (*_gradient)[id];
    _v[id] = _beta2 * _v[id] + (1 - _beta2) * gpow2;

    MatrixXd vsqrt = _v[id].cwiseSqrt();
    MatrixXd gdiv = (vsqrt + _eps[id]).cwiseInverse();

    if (_batchSize == 1) {
        (*p_connection->getWeights()) += _alpha * _m[id].cwiseProduct(gdiv);
        (*p_connection->getOutGroup()->getBias()) += _alpha * _delta[p_connection->getOutGroup()->getId()];
    }
    else {
        if (_batch < _batchSize) {
            _weightDelta[p_connection->getId()] += _alpha * _m[id].cwiseProduct(gdiv);
            _biasDelta[p_connection->getId()] += _alpha * _delta[p_connection->getOutGroup()->getId()];
        }
        else {
            (*p_connection->getWeights()) += _weightDelta[p_connection->getId()];
            (*p_connection->getOutGroup()->getBias()) += _biasDelta[p_connection->getId()];

            _weightDelta[p_connection->getId()].fill(0);
            _biasDelta[p_connection->getId()].fill(0);
        }

    }
}
