//
// Created by user on 25. 2. 2016.
//

#ifndef LIBNEURONET_QLEARNING_H
#define LIBNEURONET_QLEARNING_H


#include "GradientBase.h"

class QLearning : public GradientBase {

public:
    QLearning(NeuralNetwork *p_network, double p_gamma);

    double train(VectorXd* p_state0, VectorXd* p_action0, VectorXd* p_state1, double p_reward);
    void update(NeuralGroup* p_node);
    void updateWeights(Connection* p_connection);

    void setAlpha(double p_alpha);

private:
    double calcMaxQa(VectorXd* p_state, int p_aDim);

private:
    double  _alpha;
    double  _gamma;

    double _error;
};


#endif //LIBNEURONET_QLEARNING_H
