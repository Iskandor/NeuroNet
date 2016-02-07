#pragma once
#include "vectorN.h"
#include "IEnvironment.h"

class NBandit : public IEnvironment
{
public:
  NBandit(int p_n, int p_b);
  ~NBandit(void);

  bool evaluateAction(vectorN<double>* p_action, vectorN<double>* p_state) override;
  void updateState(vectorN<double>* p_action) override;
  void reset() override;
  double getProbability(int p_index) const;

private:
  void decodeAction(vectorN<double> *p_action, int *p_directive) const;

  vectorN<double> *_rewardDist;  
};

