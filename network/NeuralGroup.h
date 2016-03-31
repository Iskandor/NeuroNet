#pragma once

#include <vector>
#include <Eigen/Dense>
#include "filters/IFilter.h"

using namespace std;
using namespace Eigen;

class NeuralGroup
{
public:
    NeuralGroup(string p_id, int p_dim, int p_activationFunction);
    ~NeuralGroup(void);


    void fire();
    void integrate(VectorXd* p_input, MatrixXd* p_weights);
    void activate();
    void calcDerivs();

    string getId() const { return _id; };
    int getDim() const { return _dim; };

    VectorXd* getOutput() { return &_output; };
    MatrixXd* getDerivs() { return &_derivs; };
    VectorXd* getActionPotential() { return &_actionPotential; };

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

    int getActivationFunction() { return _activationFunction; };

private:
    string  _id;
    int     _dim;
    int     _activationFunction;
    bool    _valid;

    VectorXd _output;
    MatrixXd _derivs;
    VectorXd _actionPotential;

    vector<int> _inConnections;
    int _outConnection;

    vector<IFilter*> _inFilter;
    vector<IFilter*> _outFilter;
};

