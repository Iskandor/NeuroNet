//
// Created by user on 1. 5. 2016.
//

#ifndef NEURONET_CACLA_H
#define NEURONET_CACLA_H


#include "../../network/NeuralNetwork.h"
#include "ActorCritic.h"

class CACLA : public ActorCritic {
public:
    CACLA(NeuralNetwork* p_actor, NeuralNetwork* p_critic);
    ~CACLA();

    void run() override;
};


#endif //NEURONET_CACLA_H
