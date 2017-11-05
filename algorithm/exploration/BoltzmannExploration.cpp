//
// Created by user on 17. 10. 2017.
//

#include <math.h>
#include <backend/flab/RandomGenerator.h>
#include "BoltzmannExploration.h"


using namespace NeuroNet;

BoltzmannExploration::BoltzmannExploration(double p_startT, double p_endT) {
    _startT = p_startT;
    _endT = p_endT;
    _temperature = (double)pow(10, _startT);
}

int BoltzmannExploration::chooseAction(Vector* p_values) {

    double sum = 0;
    double* p = new double[p_values->size()];

    for(int i = 0; i < p_values->size(); i++) {
        sum += exp((*p_values)[i] / _temperature);
    }

    for(int i = 0; i < p_values->size(); i++) {
        p[i] = (double)exp((*p_values)[i] / _temperature) / sum;
    }

    return RandomGenerator::getInstance().choice(p, p_values->size());
}

void BoltzmannExploration::update(double f) {
    double t = linterp(_startT, _endT, f);

    _temperature = (double)pow(10, t);

    cout << _temperature << endl;
}
