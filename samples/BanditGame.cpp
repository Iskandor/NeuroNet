#include "../algorithm/IEnvironment.h"
#include "BanditGame.h"

BanditGame::BanditGame(int p_dim, int p_arm) : IEnvironment()
{
  for(int i = 0; i < p_dim; i++) {
    _bandits[i] = new NBandit(p_arm, i);
  }  
  _dim = p_dim;
  _index = 0;
  _state = new VectorXd(p_dim);
}


BanditGame::~BanditGame(void)
{
  
}

bool BanditGame::evaluateAction(VectorXd* p_action, VectorXd* p_state) {
  int index = _index;
  _bandits[index]->evaluateAction(p_action, p_state);

  index++;
  if (index == _dim) {
    index = 0;
  }
  p_state->Zero(p_state->size());
  (*p_state)[index] = 1;

  return true;
}

void BanditGame::updateState(VectorXd* p_action) {
  _bandits[_index]->updateState(p_action);
  _reward = _bandits[_index]->getReward();
  _index++;  

  if (_index == _dim) {
    _index = 0;
  }
  _state->Zero(_state->size());
  (*_state)[_index] = 1;
}

void BanditGame::reset() {
  _index = 0;
}
