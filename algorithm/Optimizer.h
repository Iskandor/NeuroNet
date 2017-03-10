//
// Created by mpechac on 10. 3. 2017.
//

#ifndef NEURONET_OPTIMIZER_H
#define NEURONET_OPTIMIZER_H

#include "GradientDescent.h"

namespace NeuroNet {

class Optimizer : public GradientDescent {

public:
    Optimizer(NeuralNetwork *p_network, const GRADIENT &p_gradient, double p_weightDecay, double p_momentum, bool p_nesterov);

    virtual double train(VectorXd *p_input, VectorXd* p_target) = 0;

protected:
    virtual void updateWeights(Connection* p_connection) = 0;

    void    calcGradient(VectorXd* p_error);
    double  calcMse(VectorXd *p_target);

    void update(NeuralGroup* p_node);
    void weightDecay(Connection* p_connection);

protected:
    GRADIENT _gradType;
    double   _naturalEpsilon;

    map<int, MatrixXd> _weightDelta;
    map<int, VectorXd> _biasDelta;
    double             _weightDecay;
    VectorXd _error;
};

}


#endif //NEURONET_OPTIMIZER_H