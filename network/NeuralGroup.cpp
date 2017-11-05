#include <memory>
#include <cmath>
#include "NeuralGroup.h"
#include "NetworkUtils.h"

using namespace std;
using namespace NeuroNet;
/**
 * NeuralGroup constructor creates layer of p_dim neurons with p_activationFunction
 * @param p_id name of layer must be unique per network
 * @param p_dim dimension of layer
 * @param p_activationFunction type of activation function
 */
NeuralGroup::NeuralGroup(string p_id, int p_dim, ACTIVATION p_activationFunction, bool p_bias)
{
    _id = p_id;
    _dim = p_dim;
    _activationFunction = p_activationFunction;

    _output = Vector::Zero(_dim);
    _ap = Vector::Zero(_dim);
    _bias = Vector::Random(_dim);
    _biasActive = p_bias;
    _derivs = Matrix::Zero(_dim, _dim);
    _valid = false;
}

NeuralGroup::NeuralGroup(NeuralGroup &p_copy) {
    _id = p_copy._id;
    _dim = p_copy._dim;
    _activationFunction = p_copy._activationFunction;

    _output = Vector::Zero(_dim);
    _ap = Vector::Zero(_dim);
    _bias = Vector(p_copy._bias);
    _biasActive = p_copy._biasActive;
    _derivs = Matrix::Zero(_dim, _dim);
    _valid = false;
}

/**
 * NeuralGroup destructor frees filters
 */
NeuralGroup::~NeuralGroup(void)
{

}

/**
 * calculates output of group and processes it by output filters
 */
void NeuralGroup::fire() {
    _valid = true;
    activate();
}

/**
 * adds input connection
 * @param p_index index of connection from connections pool
 */
void NeuralGroup::addInConnection(int p_index) {
    _inConnections.push_back(p_index);
}

/**
 * adds output connection (currently only one is possible)
 * @param p_index index of connection from connections pool
 */
void NeuralGroup::addOutConnection(int p_index) {
    _outConnections.push_back(p_index);
}

/**
 * performs product of weights and input which is stored in actionPotential vector
 * @param p_input vector of input values
 * @param p_weights matrix of input connection params
 */
void NeuralGroup::integrate(Vector* p_input, Matrix* p_weights) {
    if (_biasActive) {
        _ap += (*p_weights) * (*p_input) + _bias;
    }
    else {
        _ap += (*p_weights) * (*p_input);
    }
}

/**
 * calculates the output of layer according to activation function
 */
void NeuralGroup::activate() {
    for(auto index = 0; index < _dim; index++) {
        switch (_activationFunction) {
            case IDENTITY:
            case LINEAR:
                _output[index] = _ap[index];
                _ap[index] = 0;
                break;
            case BINARY:
                if (_ap[index] > 0) {
                    _output[index] = 1;
                    _ap[index] = 0;
                }
                else {
                    _output[index] = 0;
                }
                break;
            case SIGMOID:
                _output[index] = 1 / (1 + exp(-_ap[index]));
                _ap[index] = 0;
                break;
            case TANH:
                _output[index] = tanh(_ap[index]);
                _ap[index] = 0;
                break;
            case SOFTMAX:
                {
                double sumExp = 0;
                for(int i = 0; i < _dim; i++) {
                    sumExp += exp(_ap[i]);
                }
                _output[index] = exp(_ap[index]) / sumExp;
                _ap[index] = 0;
                }
                break;
            case SOFTPLUS:
                _output[index] = log( 1 + exp(_ap[index]));
                _ap[index] = 0;
                break;
            case RELU:
                _output[index] = max(0., _ap[index]);
                _ap[index] = 0;
                break;
        }
    }
}

/**
 * calculates derivative of the output of layer according to activation function
 */
void NeuralGroup::calcDerivs() {
    switch (_activationFunction) {
        case IDENTITY:
        case BINARY:
        case LINEAR:
            _derivs = Matrix::Identity(_dim, _dim);
            break;
        case SIGMOID:
            for(int i = 0; i < _dim; i++) {
                _derivs.set(i, i, _output[i] * (1 - _output[i]));
            }
            break;
        case TANH:
            for(int i = 0; i < _dim; i++) {
                _derivs.set(i, i, (1 - pow(_output[i], 2)));
            }
            break;
        case SOFTMAX:
            for(int i = 0; i < _dim; i++) {
                for(int j = 0; j < _dim; j++) {
                    _derivs.set(i, j, _output[i] * (NetworkUtils::kroneckerDelta(i,j) - _output[j]));
                }
            }
            break;
        case SOFTPLUS:
            for(int i = 0; i < _dim; i++) {
                _derivs.set(i, i, 1 / (1 + exp(-_output[i])));
            }
            break;
        case RELU:
            for(int i = 0; i < _dim; i++) {
                _derivs.set(i, i, (_output[i] > 0) ? 1 : 0);
            }
            break;
    }
}

json NeuralGroup::getFileData() {
    return json({{"dim", _dim}, {"actfn", _activationFunction}});
}

void NeuralGroup::setOutput(Vector *p_output) {
    _output = Vector(*p_output);
}

void NeuralGroup::setBias(Vector *p_bias) {
    _bias = Vector(*p_bias);
}
