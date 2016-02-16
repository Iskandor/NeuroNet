#include <memory>
#include <cmath>
#include "NeuralGroup.h"
#include "Define.h"

using namespace std;

NeuralGroup::NeuralGroup(int p_id, int p_dim, int p_activationFunction)
{
  _id = p_id;
  _dim = p_dim;
  _activationFunction = p_activationFunction;

  _output.init(_dim);
  _output.set(0);
  _derivs.init(_dim);
  _derivs.set(0);
  _actionPotential.init(_dim);
  _actionPotential.set(0);

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
    _outConnections.push_back(p_index);
}

void NeuralGroup::integrate(vectorN<double>* p_input, matrix2<double>* p_weights) const {
  vectorN<double> result;

  result = *p_weights * *p_input;
  _actionPotential += result;
}

/* function which should calculate the output of neuron (activation function output) according to action potential */
void NeuralGroup::activate() const {

  for(auto index = 0; index < _dim; index++) {    
    switch (_activationFunction) {
      case IDENTITY:
        _output[index] = _actionPotential[index];
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
    }
  }
}

void NeuralGroup::calcDerivs() const {
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
    }
  }
}
