#pragma once

#include <vector>
#include "../include/vectorN.h"
#include "../include/matrix2.h"

using namespace std;

class NeuralGroup
{
public:
	NeuralGroup(int p_id, int p_dim, int p_activationFunction);
	~NeuralGroup(void);


  void fire();    
  void integrate(vectorN<double>* p_input, matrix2<double>* p_weights) const;
  void activate() const;
  void calcDerivs() const;

  int getId() const
  { return _id; };
  int getDim() const
  { return _dim; };
  int getActivationFunction() const
  { return _activationFunction; };

  void    setOutput(vectorN<double>* p_vector) { _output.setVector(p_vector); };
  vectorN<double>* getOutput() { return &_output; };
  vectorN<double>* getDerivs() { return &_derivs; };

  void addOutConnection(int p_index);
  void addInConnection(int p_index);
  vector<int>* getOutConnections() { return &_outConnections; }; 
  vector<int>* getInConnections() { return &_inConnections; };
  
  bool isValid() const { return _valid; };
  void invalidate() { _valid = false; };
  void setValid() { _valid = true; };


private:
  int     _id;
  int     _dim;
  int     _activationFunction;
  bool    _valid;

  vectorN<double> _output;
  vectorN<double> _derivs;
  vectorN<double> _actionPotential;
  
  vector<int> _inConnections;
  vector<int> _outConnections;
};

