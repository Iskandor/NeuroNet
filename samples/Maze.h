#pragma once

#include "../algorithm/IEnvironment.h"

using namespace NeuroNet;

class Maze : public IEnvironment
{
public:
  Maze(int p_dim);
  ~Maze(void);

  bool evaluateAction(VectorXd* p_action, VectorXd* p_state) override;
  void updateState(VectorXd* p_action) override;
  void reset() override;
  bool isFinished() const;

  VectorXd* getPlayer();
  int getDim() const { return _dim;};

private:
  void decodeAction(VectorXd* p_action, VectorXd* p_command) const;
  bool isValidMove(double p_x, double p_y) const;

  int _dim;
  VectorXd _player;
  VectorXd _goal;
};

