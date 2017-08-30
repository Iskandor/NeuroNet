//
// Created by mpechac on 13. 3. 2017.
//

#ifndef NEURONET_ACTORLEARNING_H
#define NEURONET_ACTORLEARNING_H

#include "../optimizer/Optimizer.h"

namespace NeuroNet {

class ActorLearning {
public:
    ActorLearning(Optimizer* p_optimizer, NeuralNetwork* p_network, double p_gamma = 0.9);
    ~ActorLearning();

    double train(Vector* p_state0, int p_action, double p_value0, double p_value1, double p_reward);

    void init(double p_alpha);

private:
    NeuralNetwork*  _network;
    Optimizer*      _optimizer;
    double          _gamma;
};

}

#endif //NEURONET_ACTORLEARNING_H
