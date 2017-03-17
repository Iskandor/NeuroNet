//
// Created by mpechac on 13. 3. 2017.
//

#ifndef NEURONET_SARSA_H
#define NEURONET_SARSA_H

#include "../LearningAlgorithm.h"
#include "../optimizer/Optimizer.h"

namespace NeuroNet {

class SARSA : public LearningAlgorithm {
public:
    SARSA(Optimizer* p_optimizer, NeuralNetwork* p_network, double p_gamma);
    ~SARSA();

    double train(Vector* p_state0, int p_action0, Vector* p_state1, int p_action1, double p_reward);

    void setAlpha(double p_alpha);
    void setBatchSize(int p_batchSize);

private:
    NeuralNetwork*  _network;
    Optimizer*      _optimizer;
    double          _gamma;
};

}

#endif //NEURONET_SARSA_H
