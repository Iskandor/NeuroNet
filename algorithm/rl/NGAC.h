//
// Created by user on 11. 6. 2016.
//

#ifndef NEURONET_NGAC_H
#define NEURONET_NGAC_H

#include "../../network/NeuralNetwork.h"
#include "ActorCritic.h"

namespace NeuroNet {

class NGAC : public ActorCritic {
public:
    NGAC(NeuralNetwork* p_actor, NeuralNetwork* p_critic);
    ~NGAC();

    void run() override;
};

}


#endif //NEURONET_NGAC_H
