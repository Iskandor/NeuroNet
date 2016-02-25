#pragma once
#include <Eigen/Dense>

using namespace Eigen;

class IEnvironment
{
  public:
  IEnvironment(): _reward(0) {};

  virtual ~IEnvironment(void) {};

  virtual bool evaluateAction (VectorXd *p_action, VectorXd *p_state) = 0;
  virtual void updateState(VectorXd *p_action) = 0;
  VectorXd *getState() { return &_state; };
  virtual void reset() = 0;
  double getReward() const {return _reward;};
  
  protected:
  double _reward;
  VectorXd _state;
};
