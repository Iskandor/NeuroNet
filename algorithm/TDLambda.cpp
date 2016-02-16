#include "../network/NeuralNetwork.h"
#include "BackProp.h"
#include "TDLambda.h"

TDLambda::TDLambda(NeuralNetwork* p_network, double p_lambda, double p_gamma) : BackProp(p_network) {
  _lambda = p_lambda;
  _gamma = p_gamma;

  for(unsigned int i = 0; i < _network->getConnections()->size(); i++) {
    int id = _network->getConnections()->at(i)->getId();
    int nCols = _network->getConnections()->at(i)->getInGroup()->getDim();
    int nRows = _network->getConnections()->at(i)->getOutGroup()->getDim();
    _delta[id] = MatrixXd::Zero(nRows, nCols);
  }

  _Pt0 = VectorXd::Zero(_network->getOutputGroup()->getDim());
  _Pt1 = VectorXd::Zero(_network->getOutputGroup()->getDim());

  _input0 = new double[_network->getInputGroup()->getDim()];
  _input1 = nullptr;
}

TDLambda::~TDLambda() {
  delete[] _input0;
}

double TDLambda::train(double* p_input, double* p_target) {
  double td_error = 0;

  if (_input1 != nullptr) {
    memcpy(_input0, _input1, sizeof(double) * _network->getInputGroup()->getDim());
  }
  _input1 = p_input;
    
  // forward activation phase
  _network->setInput(_input1);
  _network->onLoop();

  _Pt0 = _Pt1;
  _Pt1 = *_network->getOutput();

  // calc TDerror
  for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
    td_error += p_target[i] + _gamma * _Pt1[i] - _Pt0[i];
  }

  // calc TD
  for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
    _gradient[_network->getOutputGroup()->getId()][i] = p_target[i] + _gamma * _Pt1[i] - _Pt0[i];
  }

  // updating phase for V(s)
  _network->setInput(_input0);
  _network->onLoop();
  backProp();

  return td_error;
}

