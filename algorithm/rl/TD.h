//
// Created by mpechac on 22. 3. 2017.
//

#ifndef NEURONET_TD_H
#define NEURONET_TD_H


#include "../optimizer/Optimizer.h"

namespace NeuroNet {

class TD {
public:
    TD(Optimizer* p_optimizer, NeuralNetwork* p_network, double p_gamma);
    ~TD();

    double train(Vector* p_state0, Vector* p_state1, double p_reward);

    void init(double p_alpha);
private:
    Optimizer*      _optimizer;
    NeuralNetwork*  _network;
    double _gamma;
};

}

#endif //NEURONET_TD_H
