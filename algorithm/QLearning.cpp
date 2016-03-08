//
// Created by Matej Pechac on 25. 2. 2016.
//

#include "QLearning.h"

QLearning::QLearning(NeuralNetwork *p_network, double p_gamma, double p_lambda) : GradientBase(p_network) {
  _gamma = p_gamma;
  _lambda = p_lambda;

  for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
    _eligTrace[it->second->getId()] = MatrixXd::Zero(it->second->getOutGroup()->getDim(), it->second->getInGroup()->getDim());
  }
}

double QLearning::train(VectorXd* p_state0, VectorXd* p_action0, VectorXd* p_state1, double p_reward) {
  _network->setInput(p_state0);
  _network->onLoop();

  double Qs0a0 = _network->getScalarOutput();
  double maxQs1a = calcMaxQa(p_state1);

  _error = p_reward + _gamma * maxQs1a - Qs0a0;

  // updating phase for Q(s,a)
  _network->setInput(p_state0);
  _network->onLoop();

  calcGradient();
  updateEligTraces();
  for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
    updateWeights(it->second);
  }

  return _error;
}

void QLearning::setAlpha(double p_alpha) {
  _alpha = p_alpha;
}

void QLearning::updateWeights(Connection *p_connection) {
  int nCols = p_connection->getInGroup()->getDim();
  int nRows = p_connection->getOutGroup()->getDim();
  MatrixXd delta(nRows, nCols);

  delta = _alpha * _error * _eligTrace[p_connection->getId()]; //_gradient[p_connection->getOutGroup()->getId()] * p_connection->getInGroup()->getOutput()->transpose();
  p_connection->getWeights()->operator-=(delta);
}

double QLearning::calcMaxQa(VectorXd* p_state) {
  double maxQa = -INFINITY;

  _network->setInput(p_state);
  _network->onLoop();
  for(int i = 0; i < _network->getOutput()->size(); i++) {
    if ((*_network->getOutput())[i] >  maxQa) {
      maxQa = (*_network->getOutput())[i];
    }
  }

  return maxQa;
}

void QLearning::updateEligTraces() {
  for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
    Connection *connection = it->second;

    _eligTrace[connection->getId()] = _gradient[connection->getOutGroup()->getId()] * connection->getInGroup()->getOutput()->transpose() + _gamma * _lambda * _eligTrace[connection->getId()];
  }
}
