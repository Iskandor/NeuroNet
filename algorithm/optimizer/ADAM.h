//
// Created by mpechac on 13. 3. 2017.
//

#ifndef NEURONET_ADAM_H
#define NEURONET_ADAM_H

#include "Optimizer.h"

namespace NeuroNet {

class ADAM : public Optimizer {
public:
    ADAM(NeuralNetwork *p_network, double p_beta1 = .9, double p_beta2 = .999, double p_epsilon = 1e-8, const GRADIENT &p_gradient = REGULAR);
    ~ADAM();

    double train(Vector *p_input, Vector* p_target);

protected:
    void updateWeights(Connection* p_connection);

private:
    double _beta1, _beta2;
    map<int, Matrix> _eps;
    map<int, Matrix> _m;
    map<int, Matrix> _v;
};

}

#endif //NEURONET_ADAM_H
