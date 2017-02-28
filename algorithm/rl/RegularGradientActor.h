//
// Created by user on 7. 5. 2016.
//

#ifndef NEURONET_REGULARGRADIENTACTOR_H
#define NEURONET_REGULARGRADIENTACTOR_H


#include "../LearningAlgorithm.h"
#include "../StochasticGradientDescent.h"

namespace NeuroNet {

    class RegularGradientActor : public StochasticGradientDescent, public LearningAlgorithm {

    public:
        RegularGradientActor(NeuralNetwork *p_network);
        virtual ~RegularGradientActor();

        void train(VectorXd* p_state0, double tdError);

    private:
        void updateWeights(Connection* p_connection);

        double _tdError;
        VectorXd  _error;
    };

}
#endif //NEURONET_REGULARGRADIENTACTOR_H
