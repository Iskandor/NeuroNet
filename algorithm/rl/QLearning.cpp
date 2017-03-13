//
// Created by Matej Pechac on 25. 2. 2016.
//

#include "QLearning.h"

using namespace NeuroNet;

QLearning::QLearning(Optimizer *p_optimizer, NeuralNetwork *p_network, double p_gamma, double p_lambda) {
    _optimizer = p_optimizer;
    _network = p_network;
    _gamma = p_gamma;
    _lambda = p_lambda;

    /*
    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
        _eligTrace[it->second->getId()] = MatrixXd::Zero(it->second->getOutGroup()->getDim(), it->second->getInGroup()->getDim());
    }
    */
}

double QLearning::train(VectorXd* p_state0, int p_action0, VectorXd* p_state1, double p_reward) {
    double mse = 0;
    double maxQs1a = calcMaxQa(p_state1);
    VectorXd target = VectorXd::Zero(_network->getOutput()->size());

    // updating phase for Q(s,a)
    _network->activate(p_state0);

    target = _network->getOutput()->replicate(1,1);
    target[p_action0] = p_reward + _gamma * maxQs1a;

    mse = _optimizer->train(p_state0, &target);

    return mse;
}

double QLearning::calcMaxQa(VectorXd* p_state) {
    double maxQa = -INFINITY;

    _network->activate(p_state);
    for(int i = 0; i < _network->getOutput()->size(); i++) {
        if ((*_network->getOutput())[i] >  maxQa) {
            maxQa = (*_network->getOutput())[i];
        }
    }

    return maxQa;
}

/*
void QLearning::updateEligTrace(Connection* p_connection) {
    _eligTrace[p_connection->getId()] = _regGradient[p_connection->getId()] + _lambda * _eligTrace[p_connection->getId()];
}
*/

QLearning::~QLearning() {
    if (_optimizer != nullptr) {
        //delete _optimizer;
    }
}


void QLearning::setAlpha(double p_alpha) {
    if (_optimizer != nullptr) {
        _optimizer->setAlpha(p_alpha);
    }
}

void QLearning::setBatchSize(int p_batchSize) {
    if (_optimizer != nullptr) {
        _optimizer->setBatchSize(p_batchSize);
    }
}
