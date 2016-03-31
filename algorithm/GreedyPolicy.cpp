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

void GreedyPolicy::getActionV(VectorXd *p_state, VectorXd *p_action) {
    double maxOutput = -INFINITY;
    int action_i = 0;

    for (int i = 0; i < p_action->size(); i++) {
        p_action->fill(0);
        (*p_action)[i] = 1;

        if (_environment->evaluateAction(p_action, p_state)) {
            double roll = rand() % 100;
            if (roll < _epsilon * 100) {
                action_i = i;
                break;
            }

            _network->activate(p_state);

            if (maxOutput < _network->getScalarOutput())
            {
                action_i = i;
                maxOutput = _network->getScalarOutput();
            }
        }
    }

    p_action->fill(0);
    (*p_action)[action_i] = 1;
}

void GreedyPolicy::getActionQ(VectorXd *p_state, VectorXd *p_action) {
    double maxOutput = -INFINITY;
    int action_i = 0;

    for (int i = 0; i < p_action->size(); i++) {
        p_action->fill(0);
        (*p_action)[i] = 1;

        if (_environment->evaluateAction(p_action, p_state)) {
            double roll = rand() % 100;
            if (roll < _epsilon * 100) {
                action_i = i;
                break;
            }

            VectorXd input(p_state->size() + p_action->size());
            input << *p_state, *p_action;
            _network->activate(&input);

            if (maxOutput < _network->getScalarOutput())
            {
                action_i = i;
                maxOutput = _network->getScalarOutput();
            }
        }
    }

    p_action->fill(0);
    (*p_action)[action_i] = 1;
}
