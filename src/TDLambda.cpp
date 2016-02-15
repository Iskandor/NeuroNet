#include "../include/NeuralNetwork.h"
#include "../include/BackProp.h"
#include "../include/TDLambda.h"

TDLambda::TDLambda(NeuralNetwork* p_network, double p_lambda, double p_gamma) : BackProp(p_network) {
  _lambda = p_lambda;
  _gamma = p_gamma;

  for(unsigned int i = 0; i < _network->getConnections()->size(); i++) {
    int id = _network->getConnections()->at(i)->getId();
    int nCols = _network->getConnections()->at(i)->getInGroup()->getDim();
    int nRows = _network->getConnections()->at(i)->getOutGroup()->getDim();
    _delta[id] = new matrix2<double>(nRows, nCols);
    _delta[id]->set(0);
  }

  _Pt0 = new vectorN<double>(_network->getOutputGroup()->getDim());
  _Pt1 = new vectorN<double>(_network->getOutputGroup()->getDim());
  _Pt0->set(0);
  _Pt1->set(0);

  _input0 = new double[_network->getInputGroup()->getDim()];
  _input1 = nullptr;
}

TDLambda::~TDLambda() {
  for(unsigned int i = 0; i < _network->getConnections()->size(); i++) {
    int id = _network->getConnections()->at(i)->getId();
    delete _delta[id];
  }

  delete _Pt0;
  delete _Pt1;
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

  _Pt0->setVector(_Pt1->getVector());
  _Pt1->setVector(_network->getOutput());

  // calc TDerror
  for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
    td_error += p_target[i] + _gamma * _Pt1->at(i) - _Pt0->at(i);
  }

  // calc TD
  for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
    _gradient[_network->getOutputGroup()->getId()][i] = p_target[i] + _gamma * _Pt1->at(i) - _Pt0->at(i);
  }

  // updating phase for V(s)
  _network->setInput(_input0);
  _network->onLoop();
  backProp();

  return td_error;
}

double TDLambda::getTDerror(double* p_input, double* p_target) const {
  double td_error = 0;

  _network->setInput(p_input);
  _network->onLoop();

  for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
    td_error += p_target[i] + _gamma * _network->getOutput()->at(i) - _Pt0->at(i);
  }

  return td_error;
}

