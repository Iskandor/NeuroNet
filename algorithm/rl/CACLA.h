//
// Created by mpechac on 22. 3. 2017.
//

#ifndef NEURONET_CACLAACTOR_H
#define NEURONET_CACLAACTOR_H


#include <Optimizer.h>

namespace NeuroNet {

class CACLA {
public:
    CACLA(Optimizer* p_optimizer, NeuralNetwork* p_network);
    ~CACLA();

    double train(Vector* p_state0, Vector* p_action, double p_delta);

    void init(double p_alpha);
private:
    Optimizer*      _optimizer;
    NeuralNetwork*  _network;
};

}

#endif //NEURONET_CACLAACTOR_H
