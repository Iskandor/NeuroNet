#pragma once
#include "../network/NeuralGroup.h"
#include "../network/Connection.h"
#include "StochasticGradientDescent.h"
#include <map>

using namespace std;

namespace NeuroNet {

class BackProp : public StochasticGradientDescent
{

public:
  explicit BackProp(NeuralNetwork* p_network, double p_weightDecay = 0, double p_momentum = 0, bool p_nesterov = false);
  virtual ~BackProp(void);

  virtual double train(double *p_input, double* p_target);

protected:
  virtual void update(NeuralGroup* p_node);
  virtual void updateWeights(Connection* p_connection);

private:
  void weightDecay(Connection* p_connection) const;

protected:
  double  _weightDecay;
  VectorXd _error;
};

}