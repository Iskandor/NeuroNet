#pragma once

#include "../algorithm/IEnvironment.h"

class Maze : public IEnvironment
{
public:
  Maze(int p_dim);
  ~Maze(void);

  bool evaluateAction(vectorN<double>* p_action, vectorN<double>* p_state) override;
  void updateState(vectorN<double>* p_action) override;
  void reset() override;
  bool isFinished() const;

  vectorN<double>* getPlayer();
  int getDim() const { return _dim;};

private:
  void decodeAction(vectorN<double>* p_action, vectorN<double>* p_command) const;
  bool isValidMove(double p_x, double p_y) const;

  int _dim;
  vectorN<double> _player;
  vectorN<double> _goal;
};

