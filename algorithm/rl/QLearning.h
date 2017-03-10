//
// Created by user on 25. 2. 2016.
//

#ifndef LIBNEURONET_QLEARNING_H
#define LIBNEURONET_QLEARNING_H

#include "../BackProp.h"
#include "../RMSProp.h"

namespace NeuroNet {

class QLearning : public RMSProp {

public:
    QLearning(NeuralNetwork *p_network, double p_gamma, double p_lambda, double p_weightDecay = 0, double p_momentum = 0, bool p_nesterov = false);

    double train(VectorXd* p_state0, int p_action0, VectorXd* p_state1, double p_reward);

private:
    void updateEligTrace(Connection* p_connection);
    double calcMaxQa(VectorXd* p_state);

private:
    double _gamma;
    double _lambda;
    VectorXd _error;
    map<int, MatrixXd> _eligTrace;
};

}
#endif //LIBNEURONET_QLEARNING_H
