#include <memory>
#include <cmath>
#include "NeuralGroup.h"
#include "Define.h"
#include "NetworkUtils.h"

using namespace std;

NeuralGroup::NeuralGroup(string p_id, int p_dim, int p_activationFunction)
{
    _id = p_id;
    _dim = p_dim;
    _activationFunction = p_activationFunction;
    _outConnection = -1;

    _output = VectorXd::Zero(_dim);
    _derivs = MatrixXd::Zero(_dim, _dim);
    _actionPotential = VectorXd::Zero(_dim);
    _valid = false;
}


NeuralGroup::~NeuralGroup(void)
{
    for(auto it = _inFilter.begin(); it != _inFilter.end(); it++) {
        delete *it;
    }

    for(auto it = _outFilter.begin(); it != _outFilter.end(); it++) {
        delete *it;
    }
}

/* calculate output of group */
void NeuralGroup::fire() {
    _valid = true;
    activate();
    processOutput(_output);
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
            case SOFTMAX:
                {
                double sumExp = 0;
                for(int i = 0; i < _dim; i++) {
                    sumExp += exp(_actionPotential[i]);
                }
                _output[index] = exp(_actionPotential[index]) / sumExp;
                _actionPotential[index] = 0;
                }
                break;
            case SOFTPLUS:
                _output[index] = log( 1 + exp(_actionPotential[index]));
                _actionPotential[index] = 0;
                break;
            case BENT:
                _output[index] = (sqrt(pow(_actionPotential[index], 2) + 1) - 1) / 2 + _actionPotential[index];
                _actionPotential[index] = 0;
                break;
        }
    }
}

void NeuralGroup::calcDerivs() {
    switch (_activationFunction) {
        case IDENTITY:
            _derivs = MatrixXd::Identity(_dim, _dim);
            break;
        case BIAS:
            _derivs = MatrixXd::Zero(_dim, _dim);
            break;
        case BINARY:
            _derivs = MatrixXd::Zero(_dim, _dim);
            break;
        case SIGMOID:
            for(int i = 0; i < _dim; i++) {
                _derivs(i,i) = _output[i] * (1 - _output[i]);
            }
            break;
        case TANH:
            for(int i = 0; i < _dim; i++) {
                _derivs(i,i) = (1 - pow(_output[i], 2));
            }
            break;
        case SOFTMAX:
            for(int i = 0; i < _dim; i++) {
                for(int j = 0; j < _dim; j++) {
                    _derivs(i,j) = _output[i] * (NetworkUtils::kroneckerDelta(i,j) - _output[j]);
                }
            }
            break;
        case SOFTPLUS:
            for(int i = 0; i < _dim; i++) {
                _derivs(i,i) = 1 / (1 + exp(-_output[i]));
            }
            break;
        case BENT:
            for(int i = 0; i < _dim; i++) {
                _derivs(i,i) = _output[i] / 2*(sqrt(pow(_output[i],2) + 1)) + 1;
            }
            break;
    }
}

void NeuralGroup::addInFilter(IFilter *p_filter) {
    _inFilter.push_back(p_filter);
}

void NeuralGroup::addOutFilter(IFilter *p_filter) {
    _outFilter.push_back(p_filter);
}

VectorXd &NeuralGroup::processInput(VectorXd &p_input) {
    for(auto it = _inFilter.begin(); it != _inFilter.end(); it++) {
        p_input = (*it)->process(&p_input);
    }
    return p_input;
}

VectorXd &NeuralGroup::processOutput(VectorXd &p_output) {
    for(auto it = _outFilter.begin(); it != _outFilter.end(); it++) {
        p_output = (*it)->process(&p_output);
    }
    return p_output;
}
