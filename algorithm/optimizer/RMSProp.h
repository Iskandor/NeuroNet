//
// Created by mpechac on 9. 3. 2017.
//

#ifndef NEURONET_RMSPROP_H
#define NEURONET_RMSPROP_H


#include "Optimizer.h"

namespace NeuroNet {

class RMSProp : public Optimizer {

public:
    RMSProp(NeuralNetwork *p_network, double p_cacheDecay = 0.9, double p_epsilon = 1e-8, const GRADIENT &p_gradient = GRADIENT::REGULAR);
    ~RMSProp();

    double train(Vector *p_input, Vector* p_target);

protected:
    void updateWeights(Connection* p_connection);

protected:
    double  _cacheDecay;
    map<int, Matrix> _eps;
    map<int, Matrix> _gradientCache;
};

}

#endif //NEURONET_RMSPROP_H
