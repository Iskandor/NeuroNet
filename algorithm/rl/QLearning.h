//
// Created by user on 25. 2. 2016.
//

#ifndef LIBNEURONET_QLEARNING_H
#define LIBNEURONET_QLEARNING_H


#include "../GradientBase.h"
#include "../LearningAlgorithm.h"

namespace NeuroNet {

class QLearning : public GradientBase, public LearningAlgorithm {

public:
    QLearning(NeuralNetwork *p_network, double p_gamma, double p_lambda);

    double train(VectorXd* p_state0, VectorXd* p_action0, VectorXd* p_state1, double p_reward);

private:
    void updateWeights(Connection* p_connection);
    void updateEligTrace(Connection* p_connection);
    double calcMaxQa(VectorXd* p_state, VectorXd* p_action);

private:
    double _gamma;
    double _lambda;
    VectorXd _error;
    map<int, MatrixXd> _eligTrace;
};

}
#endif //LIBNEURONET_QLEARNING_H
