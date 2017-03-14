//
// Created by mpechac on 9. 3. 2017.
//

#include "RMSProp.h"

using namespace NeuroNet;

RMSProp::RMSProp(NeuralNetwork *p_network, double p_cacheDecay, double p_epsilon, const GradientDescent::GRADIENT &p_gradient) : Optimizer(p_network, p_gradient) {
    int nRows;
    int nCols;

    _cacheDecay = p_cacheDecay;

    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); ++it) {
        nRows = it->second->getOutGroup()->getDim();
        nCols = it->second->getInGroup()->getDim();
        _gradientCache[it->second->getId()] = MatrixXd::Zero(nRows, nCols);
        _eps[it->second->getId()] = MatrixXd::Zero(nRows, nCols);
        _eps[it->second->getId()].setOnes();
        _eps[it->second->getId()] *= p_epsilon;
    }
}

RMSProp::~RMSProp() {

}

double RMSProp::train(VectorXd *p_input, VectorXd *p_target) {
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

void RMSProp::updateWeights(Connection *p_connection) {

    MatrixXd matrix1 = (1 - _cacheDecay) * (*_gradient)[p_connection->getId()].cwiseProduct((*_gradient)[p_connection->getId()]);
    _gradientCache[p_connection->getId()] = _cacheDecay * _gradientCache[p_connection->getId()] + matrix1;

    MatrixXd matrix2 = _gradientCache[p_connection->getId()].cwiseSqrt();
    MatrixXd matrix3 = matrix2 + _eps[p_connection->getId()];
    MatrixXd matrix4 = matrix3.cwiseInverse();

    if (_batchSize == 1) {
        (*p_connection->getWeights()) += _alpha * (*_gradient)[p_connection->getId()].cwiseProduct(matrix4);
        (*p_connection->getOutGroup()->getBias()) += _alpha * _delta[p_connection->getOutGroup()->getId()];
    }
    else {
        if (_batch < _batchSize) {
            _weightDelta[p_connection->getId()] += _alpha * (*_gradient)[p_connection->getId()].cwiseProduct(matrix4);
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