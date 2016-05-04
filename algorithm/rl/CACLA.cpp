//
// Created by user on 1. 5. 2016.
//

#include "CACLA.h"

CACLA::CACLA(NeuralNetwork *p_actor, NeuralNetwork *p_critic) : ActorCritic(p_actor, p_critic) {
  _criticLearning = new QLearning(p_critic, 0.9, 0.9);
  _actorLearning = new SimpleActor(p_actor);
}

CACLA::~CACLA() {

}


void CACLA::run() {
  VectorXd state0 = VectorXd::Zero(_environment->getState()->size());
  VectorXd state1 = VectorXd::Zero(_environment->getState()->size());
  VectorXd action0 = VectorXd::Zero(_actor->getOutput()->size());
  double reward = 0;

  state0 = *_environment->getState();
  getAction(&state0, &action0);
  _environment->updateState(&action0);
  state1 = *_environment->getState();
  reward = _environment->getReward();

  SimpleActor* actorLearning = dynamic_cast<SimpleActor *>(_actorLearning);
  QLearning* criticLearning = dynamic_cast<QLearning *>(_criticLearning);

  double tdError = criticLearning->train(&state0, &action0, &state1, reward);
  if (tdError > 0) {
    actorLearning->train(&state0, &action0);
  }
}
