//
// Created by mpechac on 9. 3. 2017.
//

#include "RMSProp.h"

using namespace NeuroNet;

RMSProp::RMSProp(NeuralNetwork *p_network, double p_cacheDecay, double p_momentum, bool p_nesterov) : GradientDescent(p_network, p_momentum, p_nesterov) {
    int nRows;
    int nCols;

    _cacheDecay = p_cacheDecay;

    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); ++it) {
        nRows = it->second->getOutGroup()->getDim();
        nCols = it->second->getInGroup()->getDim();
        _gradientCache[it->second->getId()] = MatrixXd::Zero(nRows, nCols);
        _eps[it->second->getId()] = MatrixXd::Zero(nRows, nCols);
        _eps[it->second->getId()].setOnes();
        _eps[it->second->getId()] *= 1e-4;
    }

    _error = VectorXd::Zero(p_network->getOutput()->size());
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

    calcRegGradient(&_error);
    //calcNatGradient(0.001, &_error);
    for(auto it = _groupTree.rbegin(); it != _groupTree.rend(); ++it) {
        update(*it);
    }

    updateBatch();

    return mse;
}

void RMSProp::updateBatch() {
    if (_batch < _batchSize) {
        _batch++;
    }
    else {
        _batch = 0;
    }
}

double RMSProp::calcMse(VectorXd *p_target) {
    double mse = 0;
    // calc MSE
    for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
        mse += pow((*p_target)[i] - (*_network->getOutput())[i], 2);
    }

    return mse;
}

void RMSProp::update(NeuralGroup *p_node) {
    for(vector<int>::iterator it = p_node->getInConnections()->begin(); it != p_node->getInConnections()->end(); it++) {
        updateWeights(_network->getConnections()->at(*it));
        if (_weightDecay != 0) weightDecay(_network->getConnections()->at(*it));
    }
}

void RMSProp::updateWeights(Connection *p_connection) {

    MatrixXd matrix1 = _cacheDecay * _gradientCache[p_connection->getId()];
    MatrixXd matrix2 = (1 - _cacheDecay) * _regGradient[p_connection->getId()].cwiseProduct(_regGradient[p_connection->getId()]);

    _gradientCache[p_connection->getId()] += matrix1 + matrix2;

    MatrixXd matrix3 = _gradientCache[p_connection->getId()].cwiseSqrt();
    MatrixXd matrix4 = matrix3 + _eps[p_connection->getId()];
    MatrixXd matrix5 = matrix4.cwiseInverse();

    (*p_connection->getWeights()) += _alpha * _regGradient[p_connection->getId()].cwiseProduct(matrix5);

    /*
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
    */
}

void RMSProp::weightDecay(Connection *p_connection) const {

}
