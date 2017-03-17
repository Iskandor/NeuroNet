//
// Created by mpechac on 10. 3. 2017.
//

#ifndef NEURONET_OPTIMIZER_H
#define NEURONET_OPTIMIZER_H


#include "../GradientDescent.h"

namespace NeuroNet {

class Optimizer : public GradientDescent {

public:
    Optimizer(NeuralNetwork *p_network, const GRADIENT &p_gradient = GRADIENT::REGULAR, double p_weightDecay = 0);

    virtual double train(Vector *p_input, Vector* p_target) = 0;

protected:
    virtual void updateWeights(Connection* p_connection) = 0;

    void    calcGradient(Vector* p_error);
    double  calcMse(Vector *p_target);

    void update(NeuralGroup* p_node);
    void weightDecay(Connection* p_connection);

protected:
    GRADIENT _gradType;
    double   _naturalEpsilon;

    map<int, Matrix> *_gradient;
    map<int, Matrix> _weightDelta;
    map<int, Vector> _biasDelta;
    double             _weightDecay;
    Vector _error;
};

}


#endif //NEURONET_OPTIMIZER_H
