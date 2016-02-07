#pragma once
#include "NeuralNetwork.h"
#include "NeuralGroup.cuh"
#include "Connection.h"
#include "BackProp.h"
#include <map>
#include "vectorN.h"

using namespace std;

class TDLambda : public BackProp
{

public:
  TDLambda(NeuralNetwork* p_network, double p_lambda, double p_gamma);
  ~TDLambda(void);

  double train(double *p_input, double* p_target) override;
  double getTDerror(double *p_input, double* p_target) const;

protected:
  void updateWeights(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, Connection* p_connection) override;

  double  _lambda;
  double  _gamma;

  map<int, matrix2<double>*> _delta;
  vectorN<double> *_Pt0;
  vectorN<double> *_Pt1;
  double* _input0;
  double* _input1;
  
};
