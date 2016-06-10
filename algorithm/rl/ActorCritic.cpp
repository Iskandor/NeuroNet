//
// Created by user on 10. 4. 2016.
//

#include "ActorCritic.h"
#include "QLearning.h"
#include "../../network/NetworkUtils.h"

using namespace NeuroNet;

ActorCritic::ActorCritic(NeuralNetwork *p_actor, NeuralNetwork *p_critic) {
  _actor = p_actor;
  _critic = p_critic;
  _epsilon = 0;
}

ActorCritic::~ActorCritic() {
  delete _actorLearning;
  delete _criticLearning;
}

void ActorCritic::setAlpha(double p_alpha) {
  _alpha = p_alpha;
  _criticLearning->setAlpha(_alpha);
}

void ActorCritic::setBeta(double p_beta) {
  _beta = p_beta;
  _actorLearning->setAlpha(_beta);
}

void ActorCritic::setExploration(double p_epsilon) {
  _epsilon = p_epsilon;
}

void ActorCritic::init(IEnvironment* p_environment) {
  _environment = p_environment;
  _environment->reset();
}

void ActorCritic::run() {
}

void ActorCritic::getAction(VectorXd *p_state, VectorXd *p_action) {
  int max_i = 0;
  _actor->activate(p_state);

  double roll = RandomGenerator::getInstance().random();
  if (roll < _epsilon) {
    max_i = rand() % p_action->size();
  }
  else {
    for(int i = 0; i < _actor->getOutput()->size(); i++) {
      if ((*_actor->getOutput())[max_i] < (*_actor->getOutput())[i]) {
        max_i = i;
      }
    }
  }

  p_action->fill(0);
  (*p_action)[max_i] = 1;
}
