//
// Created by user on 25. 2. 2016.
//

#ifndef LIBNEURONET_QLEARNING_H
#define LIBNEURONET_QLEARNING_H

#include <Optimizer.h>

namespace NeuroNet {

class QLearning {

public:
    QLearning(Optimizer* p_optimizer, NeuralNetwork* p_network, double p_gamma, double p_lambda);
    ~QLearning();

    double train(Vector* p_state0, int p_action0, Vector* p_state1, double p_reward);

    void init(double p_alpha);

private:
    //void updateEligTrace(Connection* p_connection);
    double calcMaxQa(Vector* p_state);

private:
    NeuralNetwork*  _network;

    double _gamma;
    double _lambda;

    Optimizer* _optimizer;

    map<int, Matrix> _eligTrace;
};

}
#endif //LIBNEURONET_QLEARNING_H
