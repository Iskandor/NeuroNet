#pragma once
#include "vectorN.h"

class IEnvironment
{
  public:
  IEnvironment(): _reward(0), _state(nullptr) {};

  virtual ~IEnvironment(void) { if (_state != nullptr) delete _state; };

  virtual bool evaluateAction (vectorN<double> *p_action, vectorN<double> *p_state) = 0;
  virtual void updateState(vectorN<double> *p_action) = 0;
  vectorN<double> *getState() const { return _state; };
  virtual void reset() = 0;
  double getReward() const {return _reward;};
  
  protected:
  double _reward;
  vectorN<double> *_state;
};
