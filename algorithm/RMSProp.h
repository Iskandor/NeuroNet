//
// Created by mpechac on 9. 3. 2017.
//

#ifndef NEURONET_RMSPROP_H
#define NEURONET_RMSPROP_H


#include "Optimizer.h"

namespace NeuroNet {

class RMSProp : public Optimizer {

public:
    RMSProp(NeuralNetwork *p_network, const GRADIENT &p_gradient = GRADIENT::REGULAR, double p_cacheDecay = 0, double p_weightDecay = 0, double p_momentum = 0, bool p_nesterov = false);
    ~RMSProp();

    double train(VectorXd *p_input, VectorXd* p_target);

protected:
    void updateWeights(Connection* p_connection);

protected:
    double  _cacheDecay;
    map<int, MatrixXd> _eps;
    map<int, MatrixXd> _gradientCache;
};

}

#endif //NEURONET_RMSPROP_H
