#include <iostream>
#include "TDLambda.h"
#include "../../network/NeuralNetwork.h"

using namespace std;
using namespace NeuroNet;

TDLambda::TDLambda(NeuralNetwork* p_network, double p_lambda, double p_gamma) : StochasticGradientDescent(p_network) {
    _lambda = p_lambda;
    _gamma = p_gamma;
    _error = VectorXd::Zero(p_network->getOutputGroup()->getDim());

    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
        int nRows = it->second->getOutGroup()->getDim();
        int nCols = it->second->getInGroup()->getDim();
        _eligTrace[it->second->getId()] = MatrixXd::Zero(nRows, nCols);
    }
}

TDLambda::~TDLambda() {
}

double TDLambda::train(VectorXd *p_state0, VectorXd *p_state1,  double reward) {
    // forward activation phase
    _network->activate(p_state0);
    _Vs0 = _network->getScalarOutput();

    _network->activate(p_state1);
    _Vs1 = _network->getScalarOutput();

    // calc TD error
    _error[0] = reward + _gamma * _Vs1 - _Vs0;

    // updating phase for V(s)
    _network->activate(p_state0);

    calcRegGradient(&_error);

    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); it++) {
        updateEligTrace(it->second);
        updateWeights(it->second);
    }

    return _error[0];
}

void TDLambda::updateWeights(Connection *p_connection) {
    int nRows = p_connection->getOutGroup()->getDim();
    int nCols = p_connection->getInGroup()->getDim();
    MatrixXd deltaW = MatrixXd::Zero(nRows, nCols);

    deltaW = _alpha * _eligTrace[p_connection->getId()];

    /*
    for(int o = 0; o < _network->getOutputGroup()->getDim(); o++) {
    for(int i = 0; i < nRows; i++) {
      for(int j = 0; j < nCols; j++) {
        deltaW(i,j) += _alpha * _error[o] * _delta[p_connection->getOutGroup()->getId()](o,i) * (*p_connection->getInGroup()->getOutput())[j];
      }
    }
    }
    */

    (*p_connection->getWeights()) += deltaW;
}

/*
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
  Connection *connection = _network->getConnection(p_group->getOutConnection());
  string outId = connection->getOutGroup()->getId();
  p_group->calcDerivs();
  _delta[p_group->getId()] = MatrixXd::Zero(_network->getOutputGroup()->getDim(), p_group->getDim());

  for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
    for(int j = 0; j < connection->getOutGroup()->getDim(); j++) {
      for(int k = 0; k < p_group->getDim(); k++) {
        _delta[p_group->getId()](i,k) += (*p_group->getDerivs())[k] * _delta[outId](i,j) * connection->getWeights()->transpose()(k, j);
      }
    }
  }
}
 */

void TDLambda::updateEligTrace(Connection *p_connection) {
    _eligTrace[p_connection->getId()] = _regGradient[p_connection->getId()] + _lambda * _eligTrace[p_connection->getId()];
}
