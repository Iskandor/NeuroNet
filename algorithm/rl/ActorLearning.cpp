//
// Created by mpechac on 13. 3. 2017.
//

#include "ActorLearning.h"

using namespace NeuroNet;

ActorLearning::ActorLearning(Optimizer *p_optimizer, NeuralNetwork *p_network, double p_gamma) {
    _optimizer = p_optimizer;
    _network = p_network;
    _gamma = p_gamma;
}


ActorLearning::~ActorLearning() {

}

double ActorLearning::train(Vector* p_state0, int p_action, double p_value0, double p_value1, double p_reward) {
    double mse = 0;

    _network->activate(p_state0);
    Vector target = Vector(*_network->getOutput());
    target[p_action] = p_reward + _gamma * p_value1 - p_value0;

    mse = _optimizer->train(p_state0, &target);

    return mse;
}

void ActorLearning::init(double p_alpha) {
    _optimizer->init(p_alpha);
}