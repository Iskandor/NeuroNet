//
// Created by user on 31. 3. 2016.
//

#ifndef NEURONET_ACTOR_H
#define NEURONET_ACTOR_H


#include "../../network/NeuralNetwork.h"
#include "../GradientBase.h"

class Actor : GradientBase {
public:
    Actor(NeuralNetwork* p_network);
    ~Actor();

    void train(VectorXd* p_state, double p_error);
    void setAlpha(double p_alpha);
    void getAction(VectorXd* p_state, VectorXd* p_action);
private:
    void updateWeights(Connection* p_connection);

private:
    double _error;
    double _alpha;
};


#endif //NEURONET_ACTOR_H
