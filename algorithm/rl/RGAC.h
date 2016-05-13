//
// Created by user on 7. 5. 2016.
//

#ifndef NEURONET_RGAC_H
#define NEURONET_RGAC_H


#include "ActorCritic.h"

namespace NeuroNet {

class RGAC : public ActorCritic {
public:
    RGAC(NeuralNetwork* p_actor, NeuralNetwork* p_critic);
    ~RGAC();

    void run() override;
};

}
#endif //NEURONET_RGAC_H
