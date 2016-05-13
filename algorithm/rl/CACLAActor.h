//
// Created by user on 10. 4. 2016.
//

#ifndef NEURONET_SIMPLEACTOR_H
#define NEURONET_SIMPLEACTOR_H


#include "../GradientBase.h"
#include "../LearningAlgorithm.h"

namespace NeuroNet {

class CACLAActor : public GradientBase, public LearningAlgorithm {
public:
    CACLAActor(NeuralNetwork* p_network);
    virtual ~CACLAActor();

    void train(VectorXd* p_state0, VectorXd* p_action0);

private:
    void updateWeights(Connection* p_connection);

    VectorXd  _error;
};

}
#endif //NEURONET_SIMPLEACTOR_H
