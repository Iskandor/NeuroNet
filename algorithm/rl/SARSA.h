//
// Created by mpechac on 31. 3. 2016.
//

#ifndef NEURONET_SARSA_H
#define NEURONET_SARSA_H

#include "../LearningAlgorithm.h"
#include "../StochasticGradientDescent.h"

namespace NeuroNet {

class SARSA : public StochasticGradientDescent, public LearningAlgorithm {

public:
    SARSA(NeuralNetwork *p_network, double p_gamma, double p_lambda);
    ~SARSA();

    double train(VectorXd* p_state0, VectorXd* p_action0, VectorXd* p_state1, VectorXd* p_action1, double p_reward);

private:
    void updateWeights(Connection* p_connection);
    void updateEligTrace(Connection* p_connection);

private:
    double _gamma;
    double _lambda;
    VectorXd _error;
    map<int, MatrixXd> _eligTrace;
};

}
#endif //NEURONET_SARSA_H
