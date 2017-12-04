//
// Created by mpechac on 13. 4. 2016.
//

#include "MSOM.h"
#include "../Define.h"

using namespace NeuroNet;

MSOM::MSOM(int p_dimInput, int p_dimX, int p_dimY, NeuralGroup::ACTIVATION p_actFunction) : SOM(p_dimInput, p_dimX, p_dimY, p_actFunction) {
    addLayer("context", p_dimInput, NeuralGroup::IDENTITY, NeuralNetwork::HIDDEN);
    addConnection("context", "lattice");
}

MSOM::~MSOM() {

}

void MSOM::train(Vector *p_input) {
    setInput(p_input);

    findWinner();
    updateWeights();
    updateContext();
}

void MSOM::activate(Vector *p_input) {
    SOM::activate(p_input);
    updateContext();
}

void MSOM::updateWeights() {
    int dimInput = getGroup("input")->getDim();
    int dimContext = getGroup("context")->getDim();
    int dimLattice = getGroup("lattice")->getDim();

    Matrix deltaW(dimLattice, dimInput);
    Matrix deltaC(dimLattice, dimContext);
    Matrix* wi = getConnection("input", "lattice")->getWeights();
    Matrix* ci = getConnection("context", "lattice")->getWeights();
    Vector* ct = getGroup("context")->getOutput();

    double theta = 0;

    for(int i = 0; i < dimLattice; i++) {
        theta = calcNeighborhood(i, GAUSSIAN);
        for(int j = 0; j < dimInput; j++) {
            deltaW.set(i, j, theta * _gamma1 * ((*_inputGroup->getOutput())[j] - wi->at(i, j)));
            deltaC.set(i, j, theta * _gamma2 * ((*ct)[j] - ci->at(i, j)));
        }
    }

    (*getConnection("input", "lattice")->getWeights()) += deltaW;
    (*getConnection("context", "lattice")->getWeights()) += deltaC;
}

void MSOM::updateContext() {
    int dim = _inputGroup->getDim();
    NeuralGroup* context = getGroup("context");
    Vector* ct = getGroup("context")->getOutput();

    Matrix* wIt = getConnection("input", "lattice")->getWeights();
    Matrix* cIt = getConnection("context", "lattice")->getWeights();

    for(int i = 0; i < ct->size(); i++) {
        (*ct)[i] = (1 - _beta) * wIt->at(_winner, i) + _beta * cIt->at(_winner, i);
    }
    //context->setOutput(ct);
}

double MSOM::calcDistance(int p_index) {
    int dim = _inputGroup->getDim();

    Matrix* xi = getConnection("input", "lattice")->getWeights();
    Matrix* ci = getConnection("context", "lattice")->getWeights();
    Vector* xt = getGroup("input")->getOutput();
    Vector* ct = getGroup("context")->getOutput();

    double dx = 0;

    for(int i = 0; i < dim; i++) {
        dx += pow((*xt)[i] - xi->at(p_index, i), 2);
    }

    double dc = 0;

    for(int i = 0; i < dim; i++) {
        dc += pow((*ct)[i] - ci->at(p_index, i), 2);
    }

    double dt = (1 - _alpha) * dx + _alpha * dc;
    return dt;
}

void MSOM::initTraining(double p_gamma1, double p_gamma2, double p_alpha, double p_beta, double p_epochs) {
    _iteration = 0;
    _qError = 0;
    _alpha = p_alpha;
    _beta = p_beta;
    _gamma1_0 = p_gamma1;
    _gamma2_0 = p_gamma2;
    _lambda = (double) (p_epochs / log(_sigma0));
    _sigma = (double) (_sigma0 * exp(-_iteration / _lambda));
    _gamma1 = (double) (_gamma1_0 * exp(-_iteration / _lambda));
    _gamma2 = (double) (_gamma2_0 * exp(-_iteration / _lambda));
}

void MSOM::initTraining(double p_alpha, double p_epochs) {
}

void MSOM::paramDecay() {
    _iteration++;
    _qError = 0;
    _sigma = (double) (_sigma0 * exp(-_iteration / _lambda));
    _gamma1 = (double) (_gamma1_0 * exp(-_iteration / _lambda));
    _gamma2 = (double) (_gamma2_0 * exp(-_iteration / _lambda));
}

void MSOM::resetContext() {
    getGroup("context")->getOutput()->fill(0);
}

json MSOM::getFileData() {
    return json({{"type", "msom"},{"dimx", _dimX}, {"dimy", _dimY}});
}
