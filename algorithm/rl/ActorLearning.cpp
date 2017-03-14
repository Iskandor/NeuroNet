//
// Created by mpechac on 13. 3. 2017.
//

#include "ActorLearning.h"

using namespace NeuroNet;

ActorLearning::ActorLearning(Optimizer *p_optimizer, NeuralNetwork *p_network, double p_gamma = 0.9) {
    _optimizer = p_optimizer;
    _network = p_network;
    _gamma = p_gamma;
}


ActorLearning::~ActorLearning() {

}

double ActorLearning::train(VectorXd* p_state0, int p_action, double p_value0, double p_value1, double p_reward) {
    double mse = 0;
    VectorXd target = VectorXd::Zero(_network->getOutput()->size());

    _network->activate(p_state0);
    target = _network->getOutput()->replicate(1,1);
    target[p_action] = p_reward + _gamma * p_value1 - p_value0;

    mse = _optimizer->train(p_state0, &target);

    return mse;
}

void ActorLearning::setAlpha(double p_alpha) {
    _optimizer->setAlpha(p_alpha);
}

void ActorLearning::setBatchSize(int p_batchSize) {
    _optimizer->setBatchSize(p_batchSize);
}
