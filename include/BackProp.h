#pragma once
#include "NeuralNetwork.h"
#include "NeuralGroup.h"
#include "Connection.h"
#include <map>

using namespace std;

class BackProp
{

public:
  BackProp(NeuralNetwork* p_network);
  ~BackProp(void);

  double train(double *p_input, double* p_target);
  void setAlpha(double p_alpha);
  void setWeightDecay(double p_weightDecay);
  void setMomentum(double p_momentum);

protected:
  void backProp();
  void backActivate(NeuralGroup* p_node);
  void updateWeights(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, Connection* p_connection);

private:
  void calcError(NeuralGroup* p_group);
  void calcDeriv(NeuralGroup* p_group);
  void calcPrevError(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, Connection* p_connection);  
  void weightDecay(Connection* p_connection) const;

protected:
  NeuralNetwork* _network;
  double  _alpha;
  double  _weightDecay;
  double  _momentum;
  double* _input;

  map<int, double*> _error;
  map<int, double*> _deriv;
  map<int, double*> _prevError;
  map<int, double*> _prevChange;
};