#include <iostream>
#include "../algorithm/IEnvironment.h"
#include "MazeOld.h"

using namespace std;

MazeOld::MazeOld(int p_dim) : IEnvironment() {
  _dim = p_dim;

  _player.resize(2);
  _goal.resize(2);
  _state = VectorXd::Zero(_dim * _dim);

  _goal[0] = _dim - 1;
  _goal[1] = _dim - 1;

  _player = _goal;
}

MazeOld::~MazeOld(void)
{
}

void MazeOld::reset() {
  while(_player == _goal) {
    //_player[0] = rand() % _dim;
    //_player[1] = rand() % _dim;
    _player[0] = 0;
    _player[1] = 0;
  }
  _state.fill(0);
  _state[_player[1] * _dim + _player[0]] = 1;
  _reward = 0;
  _failed = false;
}

bool MazeOld::isFinished() const {
  return _player == _goal;
}

bool MazeOld::isFailed() const {
  return _failed;
}

void MazeOld::updateState(VectorXd* p_action) {
  VectorXd command(4);
  decodeAction(p_action, &command);

  double newX = _player[0] + command[1] + command[3];
  double newY = _player[1] + command[0] + command[2];

  if (isValidMove(newX, newY)) {
    _player[0] = newX; // left, right
    _player[1] = newY; // up, down
    _state.fill(0);
    _state[newY * _dim + newX] = 1;
    _reward = -0.1;
  }
  else
  {
    _failed = true;
    _reward = -1;
    //_state.fill(1);

  }

  if (isFinished())
  {
    _reward = 1;
  }
}

bool MazeOld::evaluateAction(VectorXd* p_action, VectorXd* p_state) {
  VectorXd command(4);
  decodeAction(p_action, &command);

  double newX = _player[0] + command[1] + command[3];
  double newY = _player[1] + command[0] + command[2];

  p_state->fill(0);
  if (isValidMove(newX, newY)) {
    (*p_state)[newY * _dim + newX] = 1;
  }
  else {
    p_state->fill(1);
  }

  return isValidMove(newX, newY);
}

void MazeOld::decodeAction(VectorXd* p_action, VectorXd* p_command) const {
  p_command->fill(0);
  if ((*p_action)(0) == 1) { // north
    (*p_command)(0) = -1;
  }
  if ((*p_action)(1) == 1) { // east
    (*p_command)(1) = 1;
  }
  if ((*p_action)(2) == 1) { // south
    (*p_command)(2) = 1;
  }
  if ((*p_action)(3) == 1) { // west
    (*p_command)(3) = -1;
  }
}

bool MazeOld::isValidMove(double p_x, double p_y) const
{
  return p_x >= 0 && p_x < _dim && p_y >= 0 && p_y < _dim;
}

VectorXd*MazeOld::getPlayer() {
  return &_player;
}

double MazeOld::getStateValue() {
  VectorXd diffVector = _player - _goal;
  double value = diffVector.norm() == 0 ? 10 : 1 / diffVector.norm();
  return value;
}