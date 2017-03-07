//
// Created by user on 11. 6. 2016.
//

#ifndef NEURONET_NATURALGRADIENTACTOR_H
#define NEURONET_NATURALGRADIENTACTOR_H

#include "../LearningAlgorithm.h"
#include "../StochasticGradientDescent.h"

namespace NeuroNet {

    class NaturalGradientActor : public StochasticGradientDescent {

    public:
        NaturalGradientActor(NeuralNetwork *p_network);
        virtual ~NaturalGradientActor();

        void train(VectorXd* p_state0, double tdError);

    private:
        void updateWeights(Connection* p_connection);
        VectorXd  _error;
    };

}


#endif //NEURONET_NATURALGRADIENTACTOR_H
