#include <memory>
#include <cmath>
#include "NeuralGroup.h"
#include "Define.h"

using namespace std;

NeuralGroup::NeuralGroup(string p_id, int p_dim, int p_activationFunction)
{
  _id = p_id;
  _dim = p_dim;
  _activationFunction = p_activationFunction;
  _outConnection = -1;

  _output = VectorXd::Zero(_dim);
  _derivs = VectorXd::Zero(_dim);
  _actionPotential = VectorXd::Zero(_dim);

  if (_activationFunction == BIAS) {
    for(int i = 0; i < p_dim; i++) {
      _output[i] = 1;
    }
  }

  _valid = false;
}


NeuralGroup::~NeuralGroup(void)
{
}

/* calculate output of group */
void NeuralGroup::fire() {
    _valid = true;
    activate();
}

void NeuralGroup::addInConnection(int p_index) {
    _inConnections.push_back(p_index);
}

void NeuralGroup::addOutConnection(int p_index) {
    _outConnection = p_index;
}

void NeuralGroup::integrate(VectorXd* p_input, MatrixXd* p_weights) {
  _actionPotential += (*p_weights) * (*p_input);
}

/* function which should calculate the output of neuron (activation function output) according to action potential */
void NeuralGroup::activate() {

  for(auto index = 0; index < _dim; index++) {    
    switch (_activationFunction) {
      case IDENTITY:
        _output[index] = _actionPotential(index);
        _actionPotential[index] = 0;
      break;
      case BIAS:
        _output[index] = -1;
        _actionPotential[index] = 0;
      break;
      case BINARY:
        if (_actionPotential[index] > 0) {
            _output[index] = 1;
            _actionPotential[index] = 0;
        }
        else {
            _output[index] = 0;
        }
      break;
      case SIGMOID:
        _output[index] = 1 / (1 + exp(-_actionPotential[index]));
        _actionPotential[index] = 0;
      break;
      case TANH:
        _output[index] = tanh(_actionPotential[index]);
        _actionPotential[index] = 0;
      break;
    }
  }
}

void NeuralGroup::calcDerivs() {
  for(auto index = 0; index < _dim; index++) {    
    switch (_activationFunction) {
      case IDENTITY:
        _derivs[index] = 1;
      break;
      case BIAS:
        _derivs[index] = 0;
      break;
      case BINARY:
        _derivs[index] = 0;
      break;
      case SIGMOID:
        _derivs[index] = _output[index] * (1 - _output[index]);
      break;
      case TANH:
        _derivs[index] = (1 - pow(_output[index], 2));
      break;
    }
  }
}
