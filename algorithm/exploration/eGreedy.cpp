//
// Created by user on 17. 10. 2017.
//

#include <backend/flab/RandomGenerator.h>
#include "eGreedy.h"

using namespace NeuroNet;

eGreedy::eGreedy(double p_startE, double p_endE) {
    _startE = p_startE;
    _endE = p_endE;
}

int eGreedy::chooseAction(Vector *p_values) {
    int action = 0;
    double random = RandomGenerator::getInstance().random();

    if (random < _epsilon) {
        action = RandomGenerator::getInstance().random(0, p_values->size() - 1);
    }
    else {
        for(int i = 0; i < p_values->size(); i++) {
            if ((*p_values)[i] > (*p_values)[action]) {
                action = i;
            }
        }
    }

    return action;
}

void eGreedy::update(double f) {
    _epsilon = linterp(_startE, _endE, f);

    cout << _epsilon << endl;
}
