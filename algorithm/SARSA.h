//
// Created by mpechac on 31. 3. 2016.
//

#ifndef NEURONET_SARSA_H
#define NEURONET_SARSA_H


#include "GradientBase.h"

class SARSA : public GradientBase {

public:
    SARSA(NeuralNetwork *p_network, double p_gamma, double p_lambda);
    ~SARSA();

    double train(VectorXd* p_state0, VectorXd* p_action0, VectorXd* p_state1, VectorXd* p_action1, double p_reward);
    void setAlpha(double p_alpha);

private:
    void updateWeights(Connection* p_connection);
    void updateEligTrace(Connection* p_connection);

private:
    double _alpha;
    double _gamma;
    double _lambda;
    double _error;
    map<int, MatrixXd> _eligTrace;
};


#endif //NEURONET_SARSA_H
