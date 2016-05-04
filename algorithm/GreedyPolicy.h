//
// Created by mpechac on 21. 3. 2016.
//

#ifndef LIBNEURONET_GREEDYPOLICY_H
#define LIBNEURONET_GREEDYPOLICY_H


#include "../network/NeuralNetwork.h"
#include "IEnvironment.h"

using namespace Eigen;

class GreedyPolicy {
public:
    GreedyPolicy(NeuralNetwork *p_network, IEnvironment* p_environment);
    virtual ~GreedyPolicy();

    void setEpsilon(double p_value);
    void getActionV(VectorXd *p_state, VectorXd *p_action);
    void getActionQ(VectorXd *p_state, VectorXd *p_action);

private:
    NeuralNetwork* _network;
    IEnvironment* _environment;
    double _epsilon;
};


#endif //LIBNEURONET_GREEDYPOLICY_H
