//
// Created by user on 30. 8. 2017.
//

#ifndef NEURONET_DEEPQLEARNING_H
#define NEURONET_DEEPQLEARNING_H

#include <Optimizer.h>

namespace NeuroNet {

class DeepQLearning {
public:
    DeepQLearning(Optimizer* p_optimizer, NeuralNetwork* p_network, double p_gamma);
    ~DeepQLearning();

    double train(Vector* p_state0, int p_action0, Vector* p_state1, double p_reward);

    void init(double p_alpha);

private:
    double calcMaxQa(Vector* p_state);

private:
    NeuralNetwork*  _network;
    double  _alpha;
    double _gamma;

    Optimizer* _optimizer;
};

}


#endif //NEURONET_DEEPQLEARNING_H
