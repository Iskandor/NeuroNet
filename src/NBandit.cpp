#include "NBandit.h"
#include <cstdlib>
#include <Define.h>
#include <cmath>


NBandit::NBandit(int p_n, int p_b) : IEnvironment()
{
  int b = p_b;
  _rewardDist = new vectorN<double>(p_n);

  for(int x = 0; x < p_n; x++) {    
    _rewardDist->set(x, exp(-0.5 * pow(static_cast<double>(x - b), 2)) /  sqrt(2 * PI));
  }

  _state = new vectorN<double>(3);
  _state->set(0);
  _state->set(0, 1);
}


NBandit::~NBandit(void)
{
  delete _rewardDist;
}

void NBandit::decodeAction(vectorN<double>* p_action, int* p_directive) const {
  for (int i  = 0; i < p_action->size(); i++) {
    if (p_action->at(i) == 1) {
      *p_directive = i;
    }
  }
}

bool NBandit::evaluateAction(vectorN<double>* p_action, vectorN<double>* p_state) {
  return true;
}

void NBandit::updateState(vectorN<double>* p_action) {
  double roll = static_cast<double>(rand()) / RAND_MAX;
  int index = 0;

  decodeAction(p_action, &index);
  if (roll <= _rewardDist->at(index)) {
    _reward = 1;
  }
  else {
    _reward = 0;
  }
}

void NBandit::reset() {
}

double NBandit::getProbability(int p_index) const {
  return _rewardDist->at(p_index);
}
