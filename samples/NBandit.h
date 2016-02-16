#pragma once

class NBandit : public IEnvironment
{
public:
  NBandit(int p_n, int p_b);
  ~NBandit(void);

  bool evaluateAction(VectorXd* p_action, VectorXd* p_state) override;
  void updateState(VectorXd* p_action) override;
  void reset() override;
  double getProbability(int p_index) const;

private:
  void decodeAction(VectorXd *p_action, int *p_directive) const;

  VectorXd *_rewardDist;
};

