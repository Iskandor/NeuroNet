#pragma once
#include "Optimizer.h"

using namespace std;

namespace NeuroNet {

class BackProp : public Optimizer
{

public:
    BackProp(NeuralNetwork* p_network, const GRADIENT &p_gradient = GRADIENT::REGULAR, double p_weightDecay = 0, double p_momentum = 0, bool p_nesterov = false);
    virtual ~BackProp(void);

    double train(VectorXd *p_input, VectorXd* p_target);

protected:
    void updateWeights(Connection* p_connection);
};

}