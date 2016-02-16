#pragma once
#include <map>
#include "NBandit.h"

using namespace std;

class BanditGame : public IEnvironment
{
public:
  BanditGame(int p_dim, int p_arm);
  ~BanditGame(void);

  bool evaluateAction(vectorN<double>* p_action, vectorN<double>* p_state) override;
  void updateState(vectorN<double>* p_action) override;
  void reset() override;
  int getIndex() const { return _index; };
  NBandit* getBandit(int p_index) { return _bandits[p_index]; };

private:
  map<int, NBandit*>  _bandits;
  int _dim;
  int _index;

};