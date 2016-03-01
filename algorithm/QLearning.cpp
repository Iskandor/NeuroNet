//
// Created by Matej Pechac on 25. 2. 2016.
//

#include "QLearning.h"

QLearning::QLearning(NeuralNetwork *p_network, double p_gamma) : GradientBase(p_network) {
  _gamma = p_gamma;
}

double QLearning::train(VectorXd* p_state0, VectorXd* p_action0, VectorXd* p_state1, double p_reward) {
  VectorXd input(p_state0->size() + p_action0->size());
  input << *p_state0, *p_action0;

  _network->setInput(&input);
  _network->onLoop();

  double Qs0a0 = _network->getScalarOutput();
  double maxQs1a = calcMaxQa(p_state1, p_action0->size());

  _error = p_reward + _gamma * maxQs1a - Qs0a0;

  // updating phase for Q(s,a)
  _network->setInput(&input);
  _network->onLoop();

  calcGradient(_network->getOutput());
  for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
    updateWeights(it->second);
  }

  _network->onLoop();
  Qs0a0 = _network->getScalarOutput();
  maxQs1a = calcMaxQa(p_state1, p_action0->size());

  double error = p_reward + _gamma * maxQs1a - Qs0a0;
  double errDelta = fabs(_error) - fabs(error);

  return _error;
}

void QLearning::setAlpha(double p_alpha) {
  _alpha = p_alpha;
}

void QLearning::update(NeuralGroup *p_node) {
  for(vector<int>::iterator it = p_node->getInConnections()->begin(); it != p_node->getInConnections()->end(); it++) {
    updateWeights(_network->getConnections()->at(*it));
  }
}

void QLearning::updateWeights(Connection *p_connection) {
  int nCols = p_connection->getInGroup()->getDim();
  int nRows = p_connection->getOutGroup()->getDim();
  MatrixXd delta(nRows, nCols);

  delta = _alpha * _error * _gradient[p_connection->getOutGroup()->getId()] * p_connection->getInGroup()->getOutput()->transpose();
  p_connection->getWeights()->operator+=(delta);
}

double QLearning::calcMaxQa(VectorXd* p_state, int p_aDim) {
  double maxQa = -INFINITY;
  VectorXd action(p_aDim);


  for(int i = 0; i < p_aDim; i++) {
    action.fill(0);
    action[i] = 1;
    VectorXd input(p_state->size() + action.size());
    input << *p_state, action;

    _network->setInput(&input);
    _network->onLoop();
    if (_network->getScalarOutput() >  maxQa) {
      maxQa = _network->getScalarOutput();
    }
  }

  return maxQa;
}
