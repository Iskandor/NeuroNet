//
// Created by mpechac on 13. 6. 2016.
//

#include "QNatLearning.h"

using namespace NeuroNet;

QNatLearning::QNatLearning(NeuralNetwork *p_network, double p_gamma, double p_epsilon, double p_lambda) : GradientDescent(p_network) {
    _gamma = p_gamma;
    _epsilon = p_epsilon;
    _lambda = p_lambda;
    _error = VectorXd::Zero(p_network->getOutput()->size());

    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
        _eligTrace[it->second->getId()] = MatrixXd::Zero(it->second->getOutGroup()->getDim(), it->second->getInGroup()->getDim());
    }
}

double QNatLearning::train(VectorXd *p_state0, VectorXd *p_action0, VectorXd *p_state1, double p_reward) {
    VectorXd input(p_state0->size() + p_action0->size());
    input << *p_state0, *p_action0;
    _network->activate(&input);

    double Qs0a0 = _network->getScalarOutput();
    double maxQs1a = calcMaxQa(p_state1, p_action0);

    _error[0] = p_reward + _gamma * maxQs1a - Qs0a0;

    // updating phase for Q(s,a)
    _network->activate(&input);

    calcNatGradient(_epsilon, &_error);
    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
        updateEligTrace(it->second);
        updateWeights(it->second);
    }

    return _error[0];
}

void QNatLearning::updateWeights(Connection *p_connection) {
    int nCols = p_connection->getInGroup()->getDim();
    int nRows = p_connection->getOutGroup()->getDim();
    MatrixXd delta(nRows, nCols);

    delta = _alpha * _eligTrace[p_connection->getId()];
    p_connection->getWeights()->operator+=(delta);
}

void QNatLearning::updateEligTrace(Connection *p_connection) {
    _eligTrace[p_connection->getId()] = _natGradient[p_connection->getId()] + _lambda * _eligTrace[p_connection->getId()];
}

double QNatLearning::calcMaxQa(VectorXd *p_state, VectorXd *p_action) {
    double maxQa = -INFINITY;

    VectorXd action = VectorXd::Zero(p_action->size());
    for(int i = 0; i < p_action->size(); i++) {
        action.fill(0);
        action[i] = 1;
        VectorXd input(p_state->size() + p_action->size());
        input << *p_state, action;

        _network->activate(&input);

        if (_network->getScalarOutput() >  maxQa) {
            maxQa = _network->getScalarOutput();
        }
    }

    return maxQa;
}
