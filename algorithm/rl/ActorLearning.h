//
// Created by mpechac on 13. 3. 2017.
//

#ifndef NEURONET_ACTORLEARNING_H
#define NEURONET_ACTORLEARNING_H

#include "../LearningAlgorithm.h"
#include "../optimizer/Optimizer.h"

namespace NeuroNet {

class ActorLearning : public LearningAlgorithm{
public:
    ActorLearning(Optimizer* p_optimizer, NeuralNetwork* p_network, double p_gamma = 0.9);
    ~ActorLearning();

    double train(Vector* p_state0, int p_action, double p_value0, double p_value1, double p_reward);

    void setAlpha(double p_alpha);
    void setBatchSize(int p_batchSize);

private:
    NeuralNetwork*  _network;
    Optimizer*      _optimizer;
    double          _gamma;
};

}

#endif //NEURONET_ACTORLEARNING_H
