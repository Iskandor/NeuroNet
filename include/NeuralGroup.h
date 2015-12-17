#pragma once

#include <vector>
#include "cuda_runtime.h"

using namespace std;

class NeuralGroup
{
public:
	NeuralGroup(int p_id, int p_dim, int p_activationFunction);
	~NeuralGroup(void);

public:
  void init();
  void fire();
    
  cudaError_t integrate(double* p_input, double* p_weights, int p_input_dim);
  cudaError_t activate(double* p_input, const int p_activationFunction);

  int getId() { return _id; };
  int getDim() { return _dim; };
  int getActivationFunction() { return _activationFunction; };
  void    setOutput(double* p_output) { _output = p_output; };
  double* getOutput() { return _output; };

  void addOutConnection(int p_index);
  void addInConnection(int p_index);
  vector<int>* getOutConnections() { return &_outConnections; }; 
  vector<int>* getInConnections() { return &_inConnections; };
  bool isActivated() { return _activated; };
  void invalidate() { _activated = false; };


private:
  int     _id;
  int     _dim;
  int     _activationFunction;
  double* _output;
  double* _actionPotential;
  bool    _activated;
  vector<int> _inConnections;
  vector<int> _outConnections;
};

