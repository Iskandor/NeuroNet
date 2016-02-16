#pragma once

#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class NeuralGroup
{
public:
	NeuralGroup(int p_id, int p_dim, int p_activationFunction);
	~NeuralGroup(void);


  void fire();    
  void integrate(VectorXd* p_input, MatrixXd* p_weights);
  void activate();
  void calcDerivs();

  int getId() const
  { return _id; };
  int getDim() const
  { return _dim; };

  VectorXd* getOutput() { return &_output; };
  VectorXd* getDerivs() { return &_derivs; };

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

  VectorXd _output;
  VectorXd _derivs;
  VectorXd _actionPotential;
  
  vector<int> _inConnections;
  vector<int> _outConnections;
};
