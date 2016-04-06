//
// Created by user on 31. 3. 2016.
//

#include "Actor.h"

Actor::Actor(NeuralNetwork* p_network) : GradientBase(p_network) {
}

Actor::~Actor() {
}

void Actor::train(VectorXd *p_state, double p_error) {
  _error = p_error;
  _network->activate(p_state);
  calcGradient();

  for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
    updateWeights(it->second);
  }
}

void Actor::updateWeights(Connection *p_connection) {
  int nRows = p_connection->getOutGroup()->getDim();
  int nCols = p_connection->getInGroup()->getDim();
  MatrixXd deltaW = MatrixXd::Zero(nRows, nCols);

  deltaW = _alpha * _error * _gradient[p_connection->getId()];
  (*p_connection->getWeights()) += deltaW;
}

void Actor::setAlpha(double p_alpha) {
  _alpha = p_alpha;
}

void Actor::getAction(VectorXd *p_state, VectorXd *p_action) {
  _network->activate(p_state);

  int maxI = 0;
  double maxVal = -INFINITY;

  for(int i = 0; i < _network->getOutput()->size(); i++) {
    if (maxVal < (*_network->getOutput())[i]) {
      maxI = i;
      maxVal = (*_network->getOutput())[i];
    }
  }

  p_action->fill(0);
  (*p_action)[maxI] = 1;
}
