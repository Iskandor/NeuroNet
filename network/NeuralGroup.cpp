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
    _outConnection = -1;

    _output = VectorXd::Zero(_dim);
    _ap = VectorXd::Zero(_dim);
    _bias = VectorXd::Random(_dim);
    _derivs = MatrixXd::Zero(_dim, _dim);
    _valid = false;
}

/**
 * NeuralGroup destructor frees filters
 */
NeuralGroup::~NeuralGroup(void)
{
    for(auto it = _inFilter.begin(); it != _inFilter.end(); it++) {
        delete *it;
    }

    for(auto it = _outFilter.begin(); it != _outFilter.end(); it++) {
        delete *it;
    }
}

/**
 * calculates output of group and processes it by output filters
 */
void NeuralGroup::fire() {
    _valid = true;
    activate();
    processOutput(_output);
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
    _outConnection = p_index;
}

/**
 * performs product of weights and input which is stored in actionPotential vector
 * @param p_input vector of input values
 * @param p_weights matrix of input connection params
 */
void NeuralGroup::integrate(VectorXd* p_input, MatrixXd* p_weights) {
  _ap += (*p_weights) * (*p_input) + _bias;
}

/**
 * calculates the output of layer according to activation function
 */
void NeuralGroup::activate() {
    for(auto index = 0; index < _dim; index++) {
        switch (_activationFunction) {
            case IDENTITY:
            case LINEAR:
                _output[index] = _ap(index);
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
            _derivs = MatrixXd::Identity(_dim, _dim);
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
        case RELU:
            for(int i = 0; i < _dim; i++) {
                _derivs(i,i) = (_output[i] > 0) ? 1 : 0;
            }
            break;
    }
}

/**
 * adds filter into input filter queue
 */
void NeuralGroup::addInFilter(IFilter *p_filter) {
    _inFilter.push_back(p_filter);
}

/**
 * adds filter into output filter queue
 */
void NeuralGroup::addOutFilter(IFilter *p_filter) {
    _outFilter.push_back(p_filter);
}

/**
 * the input is being processed through input filter queue
 * @param p_input reference to input which is processed
 */
VectorXd &NeuralGroup::processInput(VectorXd &p_input) {
    for(auto it = _inFilter.begin(); it != _inFilter.end(); it++) {
        p_input = (*it)->process(&p_input);
    }
    return p_input;
}

/**
 * the output is being processed through output filter queue
 * @param p_output reference to input which is processed
 */
VectorXd &NeuralGroup::processOutput(VectorXd &p_output) {
    for(auto it = _outFilter.begin(); it != _outFilter.end(); it++) {
        p_output = (*it)->process(&p_output);
    }
    return p_output;
}

json NeuralGroup::getFileData() {
    return json({{"dim", _dim}, {"actfn", _activationFunction}});
}

void NeuralGroup::setOutput(VectorXd *p_output) {
    _output = p_output->replicate(1,1);
}
