//
// Created by user on 1. 5. 2016.
//

#include "LearningAlgorithm.h"

using namespace NeuroNet;

LearningAlgorithm::LearningAlgorithm(NeuralNetwork* p_network) {
  _network = p_network;
  _alpha = 0;
  _batchSize = 1;
  _batch = 0;

  int nCols;
  int nRows;

  for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); ++it) {
    nRows = it->second->getOutGroup()->getDim();
    nCols = it->second->getInGroup()->getDim();
    _weightDelta[it->second->getId()] = MatrixXd::Zero(nRows, nCols);
    _biasDelta[it->second->getId()] = VectorXd::Zero(nRows);
  }
}

LearningAlgorithm::~LearningAlgorithm() {

}

void LearningAlgorithm::setAlpha(double p_alpha) {
  _alpha = p_alpha;
}

void LearningAlgorithm::setBatchSize(int p_batchSize) {
  _batchSize = p_batchSize;
}
