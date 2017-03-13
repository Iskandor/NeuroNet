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

    double train(VectorXd *p_input, VectorXd* p_target);

protected:
    void updateWeights(Connection* p_connection);

private:
    double _beta1, _beta2;
    map<int, MatrixXd> _eps;
    map<int, MatrixXd> _m;
    map<int, MatrixXd> _v;
};

}

#endif //NEURONET_ADAM_H
