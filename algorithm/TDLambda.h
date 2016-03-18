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
  virtual void updateWeights(Connection* p_connection);


protected:
    void calcDelta();
    void deltaKernel(NeuralGroup *p_group);

    double  _alpha;
    double  _gamma;
    double  _lambda;
    VectorXd  _error;

    double _Vs0;
    double _Vs1;
    map<string, MatrixXd> _delta;

};
