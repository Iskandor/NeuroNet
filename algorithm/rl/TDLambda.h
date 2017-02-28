#pragma once
#include "../LearningAlgorithm.h"
#include "../StochasticGradientDescent.h"
#include <map>

using namespace std;

namespace NeuroNet {

class TDLambda : public StochasticGradientDescent, public LearningAlgorithm
{

public:
    TDLambda(NeuralNetwork* p_network, double p_lambda, double p_gamma);
    virtual ~TDLambda(void);

    virtual double train(VectorXd *p_state0, VectorXd *p_state1,  double reward);
    virtual void updateWeights(Connection* p_connection);


protected:
    void updateEligTrace(Connection* p_connection);
    //void calcDelta();
    //void deltaKernel(NeuralGroup *p_group);
    double  _gamma;
    double  _lambda;
    VectorXd  _error;

    double _Vs0;
    double _Vs1;
    //map<string, MatrixXd> _delta;
    map<int, MatrixXd> _eligTrace;

};

}