#include <iostream>
#include "../network/NeuralNetwork.h"
#include "BackProp.h"
#include "TDLambda.h"

using namespace std;

TDLambda::TDLambda(NeuralNetwork* p_network, double p_lambda, double p_gamma) : GradientBase(p_network) {
  _lambda = p_lambda;
  _gamma = p_gamma;
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

  calcDelta();

  for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
    updateWeights(it->second);
  }

  return _error;
}

void TDLambda::updateWeights(Connection *p_connection) {
  int nCols = p_connection->getInGroup()->getDim();
  int nRows = p_connection->getOutGroup()->getDim();
  MatrixXd delta(nRows, nCols);

  delta = _alpha * _error * *p_connection->getInGroup()->getOutput() * _delta[p_connection->getOutGroup()->getId()];

  (*p_connection->getWeights()) += delta;
}

void TDLambda::setAlpha(double p_alpha) {
  _alpha = p_alpha;
}

void TDLambda::calcDelta() {
  _network->getOutputGroup()->calcDerivs();
  _delta[_network->getOutputGroup()->getId()] = MatrixXd::Identity(_network->getOutputGroup()->getDim(), _network->getOutputGroup()->getDim());

  for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
    _delta[_network->getOutputGroup()->getId()](i, i) = (*_network->getOutputGroup()->getDerivs())[i];
  }

  for(auto it = ++_groupTree.begin(); it != _groupTree.end(); ++it) {
    deltaKernel(*it);
  }
}

void TDLambda::deltaKernel(NeuralGroup *p_group) {
  string outId;
  Connection *connection = nullptr;
  p_group->calcDerivs();
  _delta[p_group->getId()] = MatrixXd::Zero(_network->getOutputGroup()->getDim(), p_group->getDim());

  connection = _network->getConnection(p_group->getOutConnection());
  outId = connection->getOutGroup()->getId();
  MatrixXd m = p_group->getDerivs()->transpose() * _delta[outId].transpose(); // * *connection->getWeights();
  _delta[p_group->getId()] = m;
}