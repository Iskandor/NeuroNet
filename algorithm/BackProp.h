#pragma once
#include "../network/NeuralGroup.h"
#include "../network/Connection.h"
#include "GradientBase.h"
#include "LearningAlgorithm.h"
#include <map>

using namespace std;

namespace NeuroNet {

class BackProp : public GradientBase, public LearningAlgorithm
{

public:
  explicit BackProp(NeuralNetwork* p_network);
  virtual ~BackProp(void);

  virtual double train(double *p_input, double* p_target);
  void setWeightDecay(double p_weightDecay);
  void setMomentum(double p_momentum);

protected:
  virtual void backProp();
  virtual void update(NeuralGroup* p_node);
  virtual void updateWeights(Connection* p_connection);

private:
  void weightDecay(Connection* p_connection) const;

protected:
  double  _weightDecay;
  double  _momentum;
  double* _input;
  VectorXd _error;
};

}