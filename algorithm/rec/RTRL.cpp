//
// Created by mpechac on 12. 7. 2016.
//

#include "RTRL.h"

using namespace NeuroNet;

RTRL::RTRL(NeuralNetwork *p_network) : StochasticGradientDescent(p_network), LearningAlgorithm() {
    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); ++it) {
        int nRows = it->second->getInGroup()->getDim();
        int nCols = it->second->getOutGroup()->getDim();
        _derivs[it->first] = MatrixXd::Zero(nCols, nRows);
    }
}

RTRL::~RTRL() {

}

double RTRL::train(double *p_input, double *p_target) {
    return 0;
}

void RTRL::updateWeights(NeuroNet::Connection *p_connection) {

}

void RTRL::updateDerivs(Connection *p_connection) {
    for(auto it = _groupTree.begin(); it != _groupTree.end(); ++it) {
        (*it)->calcDerivs();
    }

    for(auto it = _groupTree.begin(); it != _groupTree.end(); ++it) {

    }
}
