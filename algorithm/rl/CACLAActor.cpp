//
// Created by user on 10. 4. 2016.
//

#include "CACLAActor.h"

using namespace NeuroNet;

CACLAActor::CACLAActor(NeuralNetwork *p_network) : GradientBase(p_network), LearningAlgorithm() {
  _error.resize(p_network->getOutput()->size());
}

CACLAActor::~CACLAActor() {

}

void CACLAActor::train(VectorXd *p_state0, VectorXd *p_action0) {
  _network->activate(p_state0);

  _error = *p_action0 - *_network->getOutput();

  calcRegGradient(&_error);

  for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
    updateWeights(it->second);
  }
}

void CACLAActor::updateWeights(Connection *p_connection) {
  int nCols = p_connection->getInGroup()->getDim();
  int nRows = p_connection->getOutGroup()->getDim();
  MatrixXd delta(nRows, nCols);

  delta = _alpha * _regGradient[p_connection->getId()];
  p_connection->getWeights()->operator+=(delta);
}
