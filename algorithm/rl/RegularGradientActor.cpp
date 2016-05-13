//
// Created by user on 7. 5. 2016.
//

#include "RegularGradientActor.h"

using namespace NeuroNet;

RegularGradientActor::RegularGradientActor(NeuralNetwork *p_network) : GradientBase(p_network), LearningAlgorithm() {
  _error.resize(p_network->getOutput()->size());
}

RegularGradientActor::~RegularGradientActor() {

}

void RegularGradientActor::train(VectorXd *p_state0, double tdError) {
  _network->activate(p_state0);

  _error = tdError * *_network->getOutput();
  _tdError = tdError;
  calcGradient();

  for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
    updateWeights(it->second);
  }
}

void RegularGradientActor::updateWeights(Connection *p_connection) {
  int nCols = p_connection->getInGroup()->getDim();
  int nRows = p_connection->getOutGroup()->getDim();
  MatrixXd delta(nRows, nCols);

  delta = _alpha * _tdError * _gradient[p_connection->getId()];
  p_connection->getWeights()->operator+=(delta);
}
