//
// Created by mpechac on 22. 3. 2017.
//

#include "CACLAActor.h"

using namespace NeuroNet;

CACLAActor::CACLAActor(Optimizer *p_optimizer, NeuralNetwork *p_network) : LearningAlgorithm() {
    _optimizer = p_optimizer;
    _network = p_network;
}

CACLAActor::~CACLAActor() {

}

double CACLAActor::train(Vector *p_state0, Vector *p_action, double p_delta) {
    double mse = 0;

    if (p_delta > 0) {
        Vector target = Vector(*p_action);
        mse = _optimizer->train(p_state0, &target);
    }

    return mse;
}

void CACLAActor::setAlpha(double p_alpha) {
    _optimizer->setAlpha(p_alpha);
}
