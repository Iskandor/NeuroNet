//
// Created by Matej Pechac on 25. 2. 2016.
//

#include "QLearning.h"

using namespace NeuroNet;

QLearning::QLearning(NeuralNetwork *p_network, double p_gamma, double p_lambda) : StochasticGradientDescent(p_network), LearningAlgorithm() {
    _gamma = p_gamma;
    _lambda = p_lambda;
    _error = VectorXd::Zero(p_network->getOutput()->size());

    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
        _eligTrace[it->second->getId()] = MatrixXd::Zero(it->second->getOutGroup()->getDim(), it->second->getInGroup()->getDim());
    }
}

double QLearning::train(VectorXd* p_state0, VectorXd* p_action0, VectorXd* p_state1, double p_reward) {
    VectorXd input(p_state0->size() + p_action0->size());
    input << *p_state0, *p_action0;
    _network->activate(&input);

    double Qs0a0 = _network->getScalarOutput();
    double maxQs1a = calcMaxQa(p_state1, p_action0);

    _error[0] = p_reward + _gamma * maxQs1a - Qs0a0;

    // updating phase for Q(s,a)
    _network->activate(&input);

    calcRegGradient(&_error);
    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
        updateEligTrace(it->second);
        updateWeights(it->second);
    }

    return _error[0];
}

void QLearning::updateWeights(Connection *p_connection) {
    int nCols = p_connection->getInGroup()->getDim();
    int nRows = p_connection->getOutGroup()->getDim();
    MatrixXd delta(nRows, nCols);

    delta = _alpha * _eligTrace[p_connection->getId()]; //_regGradient[p_connection->getId()];
    p_connection->getWeights()->operator+=(delta);
}

double QLearning::calcMaxQa(VectorXd* p_state, VectorXd* p_action) {
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

void QLearning::updateEligTrace(Connection* p_connection) {
    _eligTrace[p_connection->getId()] = _regGradient[p_connection->getId()] + _lambda * _eligTrace[p_connection->getId()];
}
