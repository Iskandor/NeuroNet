//
// Created by user on 1. 5. 2016.
//

#include "CACLA.h"

using namespace NeuroNet;

CACLA::CACLA(NeuralNetwork *p_actor, NeuralNetwork *p_critic) : ActorCritic(p_actor, p_critic) {
  _criticLearning = new QLearning(p_critic, 0.9, 0.9);
  _actorLearning = new CACLAActor(p_actor);
}

CACLA::~CACLA() {

}


void CACLA::run() {
  VectorXd state0 = VectorXd::Zero(_environment->getStateSize());
  VectorXd state1 = VectorXd::Zero(_environment->getStateSize());
  VectorXd action0 = VectorXd::Zero(_actor->getOutput()->size());
  double reward = 0;

  state0 = *_environment->getState();
  getAction(&state0, &action0);
  _environment->updateState(&action0);
  state1 = *_environment->getState();
  reward = _environment->getReward();

  CACLAActor* actorLearning = dynamic_cast<CACLAActor *>(_actorLearning);
  QLearning* criticLearning = dynamic_cast<QLearning *>(_criticLearning);

  double tdError = criticLearning->train(&state0, &action0, &state1, reward);
  if (tdError > 0) {
    actorLearning->train(&state0, &action0);
  }
}

void CACLA::getAction(VectorXd *p_state, VectorXd *p_action) {
  for(int i = 0; i < p_action->size(); i++) {
    double rand = RandomGenerator::getInstance().normalRandom(_epsilon);
    (*p_action)[i] = (*_actor->getOutput())[i] + rand;
  }
}