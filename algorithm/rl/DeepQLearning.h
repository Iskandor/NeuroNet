//
// Created by user on 30. 8. 2017.
//

#ifndef NEURONET_DEEPQLEARNING_H
#define NEURONET_DEEPQLEARNING_H

#include <Optimizer.h>
#include <vector>
#include <list>
#include "QLearning.h"

namespace NeuroNet {

class DeepQLearning : public QLearning {
public:
    DeepQLearning(Optimizer* p_optimizer, NeuralNetwork* p_network, double p_gamma);
    ~DeepQLearning();

    double train(Vector* p_state0, int p_action0, Vector* p_state1, double p_reward);

    void init(double p_alpha, int p_batchSize, int p_memorySize, int p_refitSize);

protected:
    double calcMaxQa(Vector* p_state);

private:
    void fillTrainingList();
    void refitTargetNetwork();

    struct BufferElem {
        BufferElem(Vector p_s0, int p_a, Vector p_s1, double p_r);

        Vector  s0;
        int     action;
        double  reward;
        Vector  s1;
    };

    list<BufferElem> _trainingList;
    vector<BufferElem> _memory;
    int _memorySize;
    int _batch, _batchSize;

    NeuralNetwork *_targetNetwork;
    int _refit, _refitSize;
};

}


#endif //NEURONET_DEEPQLEARNING_H
