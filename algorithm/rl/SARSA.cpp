//
// Created by mpechac on 13. 3. 2017.
//

#include "SARSA.h"

using namespace NeuroNet;

SARSA::SARSA(Optimizer *p_optimizer, NeuralNetwork *p_network, double p_gamma) {
    _optimizer = p_optimizer;
    _network = p_network;
    _gamma = p_gamma;
}

SARSA::~SARSA() {

}

double SARSA::train(Vector *p_state0, int p_action0, Vector *p_state1, int p_action1, double p_reward) {
    double mse = 0;

    _network->activate(p_state0);
    Vector target = Vector(*_network->getOutput());
    _network->activate(p_state1);
    target[p_action0] = p_reward + _gamma * (*_network->getOutput())[p_action1];

    mse = _optimizer->train(p_state0, &target);

    return mse;
}

void SARSA::setAlpha(double p_alpha) {
    _optimizer->setAlpha(p_alpha);
}

void SARSA::setBatchSize(int p_batchSize) {
    _optimizer->setBatchSize(p_batchSize);
}
