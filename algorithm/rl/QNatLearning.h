//
// Created by mpechac on 13. 6. 2016.
//

#ifndef NEURONET_QNATLEARNING_H
#define NEURONET_QNATLEARNING_H

#include "../LearningAlgorithm.h"
#include "../GradientDescent.h"

namespace NeuroNet {

class QNatLearning : public GradientDescent {
public:
    QNatLearning(NeuralNetwork *p_network, double p_gamma, double p_epislon, double p_lambda);

    double train(VectorXd* p_state0, VectorXd* p_action0, VectorXd* p_state1, double p_reward);

private:
    void updateWeights(Connection* p_connection);
    void updateEligTrace(Connection* p_connection);
    double calcMaxQa(VectorXd* p_state, VectorXd* p_action);

private:
    double _gamma;
    double _epsilon;
    double _lambda;
    VectorXd _error;
    map<int, MatrixXd> _eligTrace;
};

}


#endif //NEURONET_QNATLEARNING_H
