//
// Created by user on 10. 4. 2016.
//

#ifndef NEURONET_SIMPLEACTOR_H
#define NEURONET_SIMPLEACTOR_H


#include "../GradientBase.h"

class SimpleActor : public GradientBase {
public:
    SimpleActor(NeuralNetwork* p_network);
    ~SimpleActor();

    void train(VectorXd* p_state0, VectorXd* p_action0, double p_tdError);
    void setAlpha(double p_alpha);

private:
    void updateWeights(Connection* p_connection);

    double    _alpha;
    VectorXd  _error;
    double    _tdError;
};


#endif //NEURONET_SIMPLEACTOR_H
