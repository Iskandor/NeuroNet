//
// Created by mpechac on 28. 3. 2017.
//

#ifndef NEURONET_TDBP_H
#define NEURONET_TDBP_H

#include "BackProp.h"

namespace NeuroNet {

class TDBP : public BackProp {
public:
    TDBP(NeuralNetwork* p_network, double p_lambda = 0, double p_weightDecay = 0, double p_momentum = 0, bool p_nesterov = false, const GRADIENT &p_gradient = GRADIENT::REGULAR);
    virtual ~TDBP(void);

    double train(Vector *p_input, Vector* p_target);

protected:
    void updateWeights(Connection* p_connection);

    map<int, Tensor3>* calcEligTrace();

private:
    map<int, Tensor3> _e;
    map<string, Matrix> _d;
    double _lambda;
};

}

#endif //NEURONET_TDBP_H
