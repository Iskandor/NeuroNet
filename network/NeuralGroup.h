#pragma once

#include <vector>
#include "filters/IFilter.h"
#include "json.hpp"
#include "../backend/FLAB/Vector.h"

using namespace std;
using namespace FLAB;
using json = nlohmann::json;

namespace NeuroNet {

class NeuralGroup
{
public:
    enum ACTIVATION {
     IDENTITY = 0,
     BIAS = 1,
     BINARY = 2,
     SIGMOID = 3,
     TANH = 4,
     SOFTMAX = 5,
     LINEAR = 6,
     EXPONENTIAL = 7,
     SOFTPLUS = 8,
     RELU = 9,
     KEXPONENTIAL = 10,
     GAUSS = 11
    };

    NeuralGroup(string p_id, int p_dim, ACTIVATION p_activationFunction, bool p_bias);
    NeuralGroup(NeuralGroup& p_copy);
    ~NeuralGroup(void);


    void fire();
    void integrate(Vector* p_input, Matrix* p_weights);
    void activate();
    void calcDerivs();

    string getId() const { return _id; };
    int getDim() const { return _dim; };

    void setOutput(Vector* p_output);
    void setBias(Vector* p_bias);
    Vector* getOutput() { return &_output; };
    Matrix* getDerivs() { return &_derivs; };
    Vector* getBias() { return &_bias; };

    void addOutConnection(int p_index);
    void addInConnection(int p_index);
    vector<int>* getOutConnection() { return &_outConnections; };
    vector<int>* getInConnections() { return &_inConnections; };

    bool isValid() const { return _valid; };
    void invalidate() { _valid = false; };
    void setValid() { _valid = true; };

    ACTIVATION getActivationFunction() { return _activationFunction; };

    json getFileData();

private:
    string  _id;
    int     _dim;
    ACTIVATION _activationFunction;
    bool    _valid;

    Vector _output;
    Matrix _derivs;
    Vector _ap;
    bool _biasActive;
    Vector _bias;

    vector<int> _inConnections;
    vector<int> _outConnections;
};

}
