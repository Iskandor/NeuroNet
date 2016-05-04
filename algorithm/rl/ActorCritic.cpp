//
// Created by user on 10. 4. 2016.
//

#include "ActorCritic.h"
#include "QLearning.h"

ActorCritic::ActorCritic(NeuralNetwork *p_actor, NeuralNetwork *p_critic) {
  _actor = p_actor;
  _critic = p_critic;
  //_actorLearning = new SimpleActor(_actor);
  //_criticLearning = new QLearning(_critic, 0.9, 0.9);
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
  /*
  VectorXd state0 = VectorXd::Zero(_environment->getState()->size());
  VectorXd state1 = VectorXd::Zero(_environment->getState()->size());
  VectorXd action0 = VectorXd::Zero(_actor->getOutput()->size());
  double reward = 0;

  state0 = *_environment->getState();
  getAction(&state0, &action0);
  _environment->updateState(&action0);
  state1 = *_environment->getState();
  reward = _environment->getReward();

  double tdError = _criticLearning->train(&state0, &action0, &state1, reward);
  _actorLearning->train(&state0, &action0, tdError);
  */
}

void ActorCritic::getAction(VectorXd *p_state, VectorXd *p_action) {
  int max_i = 0;
  _actor->activate(p_state);

  double roll = (double)rand() / RAND_MAX;
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
