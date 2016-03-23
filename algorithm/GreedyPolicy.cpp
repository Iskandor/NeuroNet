//
// Created by mpechac on 21. 3. 2016.
//

#include "GreedyPolicy.h"

GreedyPolicy::GreedyPolicy(NeuralNetwork *p_network, IEnvironment *p_environment) {
    _network = p_network;
    _environment = p_environment;
    _epsilon = 0;
}

GreedyPolicy::~GreedyPolicy() {

}

void GreedyPolicy::setEpsilon(double p_value) {
    _epsilon = p_value;
}

void GreedyPolicy::getAction(VectorXd &p_action, int p_stateDim) {
    double maxOutput = -INFINITY;
    int action_i = 0;
    VectorXd state(p_stateDim);

    for (int i = 0; i < p_action.size(); i++) {
        p_action.fill(0);
        p_action[i] = 1;

        if (_environment->evaluateAction(&p_action, &state)) {
            double roll = static_cast<double>(rand()) / RAND_MAX;

            if (roll < _epsilon) {
                action_i = i;
                break;
            }

            _network->activate(&state);

            if (maxOutput < _network->getScalarOutput())
            {
                action_i = i;
                maxOutput = _network->getScalarOutput();
            }
        }
    }

    p_action.fill(0);
    p_action[action_i] = 1;
}
