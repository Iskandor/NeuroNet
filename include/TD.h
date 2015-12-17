#include "NeuralNetwork.h"
#include "NeuralGroup.h"
#include "Connection.h"

#include <matrix3.h>
#include <map>

using namespace std;

class TD
{
public:
  TD(NeuralNetwork* p_network);
  ~TD(void);

  double train(double* p_input, double* p_target);
  void setAlpha(double p_alpha);
  void setLambda(double p_lambda);

private:
  void TDRec(NeuralGroup* p_node);
  void updateWeights(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, Connection* p_connection);
  void updateEligibility(Connection* p_connection);

  void calcDelta(Connection* p_connection);
  void calcEligDelta(Connection* p_connection);
  void calcDeriv(NeuralGroup* p_group);
  void calcError() const;

private:
  NeuralNetwork* _network;
  double  _alpha;
  double  _lambda;
  int _t;

  double*  _outputT;
  double*  _error; 

  map<int, double*> _deriv;
  map<int, matrix2<double>*> _delta;
  map< int, map<int, matrix2<double>*>> _eligDelta;
  map<int, matrix3<double>*> _eligibility;
};