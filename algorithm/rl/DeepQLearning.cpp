//
// Created by user on 30. 8. 2017.
//

#include "DeepQLearning.h"

using namespace NeuroNet;

DeepQLearning::DeepQLearning(NeuroNet::Optimizer *p_optimizer, NeuralNetwork* p_network, double p_gamma) {
    _optimizer = p_optimizer;
    _network = p_network;
    _gamma = p_gamma;
}

DeepQLearning::~DeepQLearning() {

}

double DeepQLearning::train(Vector *p_state0, int p_action0, Vector *p_state1, double p_reward) {
    return 0;
}

void DeepQLearning::init(double p_alpha) {
    _optimizer->init(p_alpha);
}

double DeepQLearning::calcMaxQa(Vector *p_state) {
    return 0;
}
