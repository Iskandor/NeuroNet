//
// Created by user on 10. 4. 2016.
//

#include "SimpleActor.h"

SimpleActor::SimpleActor(NeuralNetwork *p_network) : GradientBase(p_network), LearningAlgorithm() {
  _error.resize(p_network->getOutput()->size());
  _alpha = 0;
}

SimpleActor::~SimpleActor() {

}

void SimpleActor::train(VectorXd *p_state0, VectorXd *p_action0) {
  _network->activate(p_state0);

  _error = *p_action0 - *_network->getOutput();

  calcGradient(&_error);

  for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
    updateWeights(it->second);
  }
}

void SimpleActor::updateWeights(Connection *p_connection) {
  int nCols = p_connection->getInGroup()->getDim();
  int nRows = p_connection->getOutGroup()->getDim();
  MatrixXd delta(nRows, nCols);

  delta = _alpha * _gradient[p_connection->getId()];
  p_connection->getWeights()->operator+=(delta);
}
