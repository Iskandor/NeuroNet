//
// Created by user on 10. 4. 2016.
//

#include "SimpleActor.h"

SimpleActor::SimpleActor(NeuralNetwork *p_network) : GradientBase(p_network) {
  _error.resize(p_network->getOutput()->size());
}

SimpleActor::~SimpleActor() {

}

void SimpleActor::train(VectorXd *p_state0, VectorXd *p_action0, double p_tdError) {
  _network->activate(p_state0);

  _tdError = p_tdError;
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

  delta = _alpha * _tdError * _gradient[p_connection->getId()];
  p_connection->getWeights()->operator+=(delta);
}

void SimpleActor::setAlpha(double p_alpha) {
  _alpha = p_alpha;
}
