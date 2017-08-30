//
// Created by mpechac on 22. 3. 2017.
//

#include "TD.h"

using namespace NeuroNet;

TD::TD(NeuroNet::Optimizer *p_optimizer, NeuroNet::NeuralNetwork *p_network, double p_gamma) {
    _optimizer = p_optimizer;
    _network = p_network;
    _gamma = p_gamma;
}

TD::~TD() {

}

double TD::train(Vector *p_state0, Vector *p_state1, double p_reward) {
    double mse = 0;

    _network->activate(p_state1);
    double Vs1 = (*_network->getOutput())[0];

    Vector target(_network->getOutput()->size());
    target[0] = p_reward + _gamma * Vs1;

    mse = _optimizer->train(p_state0, &target);

    return mse;
}

void TD::init(double p_alpha) {
    _optimizer->init(p_alpha);
}
