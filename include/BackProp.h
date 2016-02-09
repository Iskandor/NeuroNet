#pragma once
#include "NeuralNetwork.h"
#include "NeuralGroup.h"
#include "Connection.h"
#include <map>

using namespace std;

class BackProp
{

public:
  explicit BackProp(NeuralNetwork* p_network);
  virtual ~BackProp(void);

  virtual double train(double *p_input, double* p_target);
  void setAlpha(double p_alpha);
  void setWeightDecay(double p_weightDecay);
  void setMomentum(double p_momentum);

protected:
  virtual void backProp();
  virtual void update(NeuralGroup* p_node);
  virtual void updateWeights(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, Connection* p_connection);

private:
  void bfsCreate();
  void bfsRecursive(NeuralGroup* p_node);
  void calcGradient(NeuralGroup* p_group);  
  void weightDecay(Connection* p_connection) const;

protected:
  NeuralNetwork* _network;
  double  _alpha;
  double  _weightDecay;
  double  _momentum;
  double* _input;

  map<int, vectorN<double>> _gradient;
  vector<NeuralGroup*> _bfsTree;
};