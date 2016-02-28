#include <iostream>
#include "../network/NeuralNetwork.h"
#include "BackProp.h"
#include "TDLambda.h"

using namespace std;

TDLambda::TDLambda(NeuralNetwork* p_network, double p_lambda, double p_gamma) : GradientBase(p_network) {
  _lambda = p_lambda;
  _gamma = p_gamma;

  Connection* connection = nullptr;

  for(auto it = _network->getConnections()->rbegin(); it != _network->getConnections()->rend(); ++it) {
    connection = it->second;
    _gradientT1[connection->getId()] = MatrixXd::Zero(connection->getWeights()->rows(), connection->getWeights()->cols());
  }
}

TDLambda::~TDLambda() {
}

double TDLambda::train(VectorXd *p_state0, VectorXd *p_state1,  double reward) {
  // forward activation phase
  _network->setInput(p_state0);
  _network->onLoop();
  _Vs0 = _network->getScalarOutput();

  _network->setInput(p_state1);
  _network->onLoop();
  _Vs1 = _network->getScalarOutput();

  // calc TD error
  _error = reward + _gamma * _Vs1 - _Vs0;

  // updating phase for V(s)
  _network->setInput(p_state0);
  _network->onLoop();
  calcGradient(_network->getOutput());

  for(auto it = _groupTree.rbegin(); it != _groupTree.rend(); ++it) {
    update(*it);
  }

  // TEST
  _network->setInput(p_state0);
  _network->onLoop();
  _Vs0 = _network->getScalarOutput();

  _network->setInput(p_state1);
  _network->onLoop();
  _Vs1 = _network->getScalarOutput();

  // calc TD error
  double error = reward + _gamma * _Vs1 - _Vs0;
  double errDelta = _error - error;

  if (errDelta < 0) {
    return errDelta;
  }

  return errDelta;
}

void TDLambda::update(NeuralGroup* p_node) {
  for(vector<int>::iterator it = p_node->getInConnections()->begin(); it != p_node->getInConnections()->end(); it++) {
    updateWeights(_network->getConnections()->at(*it));
  }
}

void TDLambda::updateWeights(Connection *p_connection) {
  int nCols = p_connection->getInGroup()->getDim();
  int nRows = p_connection->getOutGroup()->getDim();
  MatrixXd delta(nRows, nCols);

  delta = _alpha * _error * _gradient[p_connection->getId()];

  (*p_connection->getWeights()) += delta;
}

void TDLambda::setAlpha(double p_alpha) {
  _alpha = p_alpha;
}

void TDLambda::calcGradient(VectorXd *p_error) {
  GradientBase::calcGradient(p_error);

  for(auto it = _network->getConnections()->rbegin(); it != _network->getConnections()->rend(); ++it) {
    _gradientT1[it->second->getId()] += _lambda * _gradient[it->second->getId()];
  }
}
