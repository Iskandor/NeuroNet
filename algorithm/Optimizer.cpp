//
// Created by mpechac on 10. 3. 2017.
//

#include "Optimizer.h"

using namespace NeuroNet;

Optimizer::Optimizer(NeuralNetwork *p_network, const GradientDescent::GRADIENT &p_gradient, double p_weightDecay) : GradientDescent(p_network) {
    _gradType = p_gradient;
    _network = p_network;
    _naturalEpsilon = 1e-3;
    _weightDecay = p_weightDecay;
    _error = VectorXd::Zero(p_network->getOutput()->size());

    int nCols;
    int nRows;

    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); ++it) {
        nRows = it->second->getOutGroup()->getDim();
        nCols = it->second->getInGroup()->getDim();
        _weightDelta[it->second->getId()] = MatrixXd::Zero(nRows, nCols);
        _biasDelta[it->second->getId()] = VectorXd::Zero(nRows);
    }
}

double Optimizer::calcMse(VectorXd *p_target) {
    double mse = 0;
    // calc MSE
    for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
        mse += pow((*p_target)[i] - (*_network->getOutput())[i], 2);
    }

    return mse;
}

void Optimizer::update(NeuralGroup *p_node) {
    for(vector<int>::iterator it = p_node->getInConnections()->begin(); it != p_node->getInConnections()->end(); it++) {
        updateWeights(_network->getConnections()->at(*it));
        if (_weightDecay != 0) weightDecay(_network->getConnections()->at(*it));
    }
}

void Optimizer::weightDecay(Connection *p_connection) {
    *p_connection->getWeights() *= (1 - _weightDecay);
}

void Optimizer::calcGradient(VectorXd* p_error) {
    switch(_gradType) {
        case GRADIENT::REGULAR:
            _gradient = calcRegGradient(p_error);
            break;
        case GRADIENT::NATURAL:
            _gradient = calcNatGradient(_naturalEpsilon, p_error);
            break;
        default:
            assert("No gradient type defined!");
    }

}
