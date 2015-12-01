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

private:
  void calcError(NeuralGroup* p_group);
  void calcDeriv(NeuralGroup* p_group);
  void calcPrevError(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, Connection* p_connection);
  void updateWeights(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, Connection* p_connection);
  void backProp();
  void backActivate(NeuralGroup* p_node);

private:
  NeuralNetwork* _network;
  double _alpha;
  double* _input;

  map<int, double*> _error;
  map<int, double*> _deriv;
  map<int, double*> _prevError;
};