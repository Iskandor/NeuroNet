#pragma once

#include <vector>
#include <Eigen/Dense>
#include "filters/IFilter.h"
#include "json.hpp"

using namespace std;
using namespace Eigen;
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
     RELU = 9
    };

    NeuralGroup(string p_id, int p_dim, ACTIVATION p_activationFunction, bool p_bias = true);
    ~NeuralGroup(void);


    void fire();
    void integrate(VectorXd* p_input, MatrixXd* p_weights);
    void activate();
    void calcDerivs();

    string getId() const { return _id; };
    int getDim() const { return _dim; };

    VectorXd* getOutput() { return &_output; };
    MatrixXd* getDerivs() { return &_derivs; };
    VectorXd* getBias() { return &_bias; };

    void addOutConnection(int p_index);
    void addInConnection(int p_index);
    int getOutConnection() { return _outConnection; };
    vector<int>* getInConnections() { return &_inConnections; };

    void addInFilter(IFilter* p_filter);
    void addOutFilter(IFilter* p_filter);
    VectorXd& processInput(VectorXd& p_input);
    VectorXd& processOutput(VectorXd& p_output);

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

    VectorXd _output;
    MatrixXd _derivs;
    VectorXd _ap;
    VectorXd _bias;

    vector<int> _inConnections;
    int _outConnection;

    vector<IFilter*> _inFilter;
    vector<IFilter*> _outFilter;
};

}
