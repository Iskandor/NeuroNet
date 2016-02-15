#include "../include/Maze.h"
#include <cstdlib>

Maze::Maze(int p_dim) : IEnvironment() {
  _dim = p_dim;

  _player.init(2);
  _goal.init(2);
  _state = new vectorN<double>(_dim * _dim);

  _goal[0] = _dim - 1;
  _goal[1] = _dim - 1;

  _player = _goal;
}

Maze::~Maze(void)
{
}

void Maze::reset() {
  while(_player == _goal) {
    _player[0] = rand() % _dim;
    _player[1] = rand() % _dim;
    //_player[0] = 0;
    //_player[1] = 0;
  }
  _state->set(0);
  _state->set(_player[1] * _dim + _player[0], 1);
  _reward = 0;
}

bool Maze::isFinished() const {
  return _player == _goal;
}

void Maze::updateState(vectorN<double>* p_action) {
  vectorN<double> command(4);
  decodeAction(p_action, &command);

  double newX = _player[0] + command[0] + command[2];
  double newY = _player[1] + command[1] + command[3];

  if (isValidMove(newX, newY)) {
    _player[0] = newX; // left, right
    _player[1] = newY; // up, down
    _reward = 0;

    _state->set(0);
    _state->set(newY * _dim + newX, 1);
  }
  else
  {
    _state->set(1);
    _reward = -1;
  }

  if (isFinished())
  {
    _reward = 1;
  }
}

bool Maze::evaluateAction(vectorN<double>* p_action, vectorN<double>* p_state) {
  vectorN<double> command(4);
  decodeAction(p_action, &command);

  double newX = _player[0] + command[0] + command[2];
  double newY = _player[1] + command[1] + command[3];
  
  p_state->set(0);
  if (isValidMove(newX, newY)) {
    p_state->set(newY * _dim + newX, 1);
  }
  else {
    p_state->set(1);
  }

  return isValidMove(newX, newY);
}

void Maze::decodeAction(vectorN<double>* p_action, vectorN<double>* p_command) const {
  p_command->set(0);
  if (p_action->at(0) == 1) {
    p_command->set(0, 1);
  }
  if (p_action->at(1) == 1) {
    p_command->set(1, 1);
  }
  if (p_action->at(2) == 1) {
    p_command->set(2, -1);
  }
  if (p_action->at(3) == 1) {
    p_command->set(3, -1);
  }
}

bool Maze::isValidMove(double p_x, double p_y) const
{
  return p_x >= 0 && p_x < _dim && p_y >= 0 && p_y < _dim;
}

vectorN<double>* Maze::getPlayer() {
  return &_player;
}
