#pragma once
#include "BackProp.h"
#include <map>

using namespace std;

class TDLambda : public BackProp
{

public:
  TDLambda(NeuralNetwork* p_network, double p_lambda, double p_gamma);
  ~TDLambda(void);

  double train(double *p_input, double* p_target) override;
  double getTDerror(double *p_input, double* p_target) const;

protected:
  double  _lambda;
  double  _gamma;

  map<int, MatrixXd> _delta;
  VectorXd _Pt0;
  VectorXd _Pt1;
  double* _input0;
  double* _input1;
  
};
