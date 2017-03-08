//
// Created by user on 11. 6. 2016.
//

#include "NGAC.h"
#include "NaturalGradientActor.h"

using namespace NeuroNet;

NGAC::NGAC(NeuralNetwork *p_actor, NeuralNetwork *p_critic) : ActorCritic(p_actor, p_critic) {
    _actorLearning = new NaturalGradientActor(p_actor);
    _criticLearning = new QLearning(p_critic, 0.9, 0.9);
}

NGAC::~NGAC() {

}

void NGAC::run() {
    VectorXd state0 = VectorXd::Zero(_environment->getState()->size());
    VectorXd state1 = VectorXd::Zero(_environment->getState()->size());
    VectorXd action0 = VectorXd::Zero(_actor->getOutput()->size());
    double reward = 0;

    state0 = *_environment->getState();
    getAction(&state0, &action0);
    _environment->updateState(&action0);
    state1 = *_environment->getState();
    reward = _environment->getReward();

    NaturalGradientActor* actorLearning = dynamic_cast<NaturalGradientActor *>(_actorLearning);
    QLearning* criticLearning = dynamic_cast<QLearning *>(_criticLearning);

    //double tdError = criticLearning->train(&state0, &action0, &state1, reward);
    //actorLearning->train(&state0, tdError);
}
