#pragma once
#include "GradientBase.h"
#include <map>

using namespace std;

class TDLambda : public GradientBase
{

public:
  TDLambda(NeuralNetwork* p_network, double p_lambda, double p_gamma);
  ~TDLambda(void);

  virtual double train(VectorXd *p_state0, VectorXd *p_state1,  double reward);
  void setAlpha(double p_alpha);
  virtual void update(NeuralGroup* p_node);
  virtual void updateWeights(Connection* p_connection);

protected:
  void calcGradient(VectorXd* p_error) override;

  double  _alpha;
  double  _gamma;
  double  _lambda;

  double _error;
  double _Vs0;
  double _Vs1;
  map<int, MatrixXd> _gradientT1;
  
};
