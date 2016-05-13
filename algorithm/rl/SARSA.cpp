//
// Created by mpechac on 31. 3. 2016.
//

#include "SARSA.h"

using namespace NeuroNet;

SARSA::SARSA(NeuralNetwork *p_network, double p_gamma, double p_lambda) : GradientBase(p_network) {
    _gamma = p_gamma;
    _lambda = p_lambda;
    _error = VectorXd::Zero(p_network->getOutput()->size());

    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
        _eligTrace[it->second->getId()] = MatrixXd::Zero(it->second->getOutGroup()->getDim(), it->second->getInGroup()->getDim());
    }
}

SARSA::~SARSA() {
}

double SARSA::train(VectorXd *p_state0, VectorXd *p_action0, VectorXd *p_state1, VectorXd *p_action1, double p_reward) {
    VectorXd input(p_state0->size() + p_action0->size());
    input << *p_state0, *p_action0;
    _network->activate(&input);
    double Qs0a0 = _network->getScalarOutput();

    input << *p_state1, *p_action1;
    _network->activate(&input);
    double Qs1a1 = _network->getScalarOutput();

    _error[0] = p_reward + _gamma * Qs1a1 - Qs0a0;

    // updating phase for Q(s,a)
    input << *p_state0, *p_action0;
    _network->activate(&input);

    calcGradient(&_error);
    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
        updateEligTrace(it->second);
        updateWeights(it->second);
    }

    return _error[0];
}

void SARSA::updateWeights(Connection *p_connection) {
    int nCols = p_connection->getInGroup()->getDim();
    int nRows = p_connection->getOutGroup()->getDim();
    MatrixXd delta(nRows, nCols);

    delta = _alpha * _eligTrace[p_connection->getId()];
    p_connection->getWeights()->operator+=(delta);
}

void SARSA::updateEligTrace(Connection *p_connection) {
    _eligTrace[p_connection->getId()] = _gradient[p_connection->getId()] + _lambda * _eligTrace[p_connection->getId()];
}
