//
// Created by user on 7. 5. 2016.
//

#include "RGAC.h"
#include "RegularGradientActor.h"

using namespace NeuroNet;

RGAC::RGAC(NeuralNetwork *p_actor, NeuralNetwork *p_critic) : ActorCritic(p_actor, p_critic) {
  _actorLearning = new RegularGradientActor(p_actor);
  _criticLearning = new QLearning(p_critic, 0.9, 0.9);
}

RGAC::~RGAC() {

}

void RGAC::run() {
  VectorXd state0 = VectorXd::Zero(_environment->getState()->size());
  VectorXd state1 = VectorXd::Zero(_environment->getState()->size());
  VectorXd action0 = VectorXd::Zero(_actor->getOutput()->size());
  double reward = 0;

  state0 = *_environment->getState();
  getAction(&state0, &action0);
  _environment->updateState(&action0);
  state1 = *_environment->getState();
  reward = _environment->getReward();

  RegularGradientActor* actorLearning = dynamic_cast<RegularGradientActor *>(_actorLearning);
  QLearning* criticLearning = dynamic_cast<QLearning *>(_criticLearning);

  double tdError = criticLearning->train(&state0, &action0, &state1, reward);
  actorLearning->train(&state0, tdError);
}
