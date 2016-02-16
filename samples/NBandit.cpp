#include <cstdlib>
#include <cmath>
#include "../algorithm/IEnvironment.h"
#include "NBandit.h"
#include "../network/Define.h"

NBandit::NBandit(int p_n, int p_b) : IEnvironment()
{
  int b = p_b;
  _rewardDist = new VectorXd(p_n);

  for(int x = 0; x < p_n; x++) {    
    (*_rewardDist)[x] =  exp(-0.5 * pow(static_cast<double>(x - b), 2)) /  sqrt(2 * PI);
  }

  _state = new VectorXd(3);
  _state->Zero(_state->size());
  (*_state)(0) = 1;
}


NBandit::~NBandit(void)
{
  delete _rewardDist;
}

void NBandit::decodeAction(VectorXd* p_action, int* p_directive) const {
  for (int i  = 0; i < p_action->size(); i++) {
    if ((*p_action)(i) == 1) {
      *p_directive = i;
    }
  }
}

bool NBandit::evaluateAction(VectorXd* p_action, VectorXd* p_state) {
  return true;
}

void NBandit::updateState(VectorXd* p_action) {
  double roll = static_cast<double>(rand()) / RAND_MAX;
  int index = 0;

  decodeAction(p_action, &index);
  if (roll <= (*_rewardDist)(index)) {
    _reward = 1;
  }
  else {
    _reward = 0;
  }
}

void NBandit::reset() {
}

double NBandit::getProbability(int p_index) const {
  return (*_rewardDist)(p_index);
}
