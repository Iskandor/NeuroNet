//
// Created by user on 25. 2. 2016.
//

#ifndef LIBNEURONET_QLEARNING_H
#define LIBNEURONET_QLEARNING_H


#include "GradientBase.h"

class QLearning : public GradientBase {

public:
    QLearning(NeuralNetwork *p_network, double p_gamma, double p_lambda);

    double train(VectorXd* p_state0, VectorXd* p_action0, VectorXd* p_state1, double p_reward);
    void setAlpha(double p_alpha);

private:
    void updateWeights(Connection* p_connection);
    void updateEligTraces();
    double calcMaxQa(VectorXd* p_state, VectorXd* p_action);

private:
    double _alpha;
    double _gamma;
    double _lambda;
    double _error;
    map<int, MatrixXd> _eligTrace;
};


#endif //LIBNEURONET_QLEARNING_H
