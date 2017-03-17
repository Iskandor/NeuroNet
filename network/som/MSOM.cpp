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
    onLoop();

    findWinner();
    updateWeights();
    updateContext();
}

void MSOM::activate(Vector *p_input) {
    SOM::activate(p_input);
}

void MSOM::updateWeights() {
    Matrix deltaW(getGroup("lattice")->getDim(), getGroup("input")->getDim());
    double theta = 0;

    for(int i = 0; i < getGroup("lattice")->getDim(); i++) {
        theta = calcNeighborhood(i, GAUSSIAN);
        Vector wi = getConnection("input", "lattice")->getWeights()->row(i);
        deltaW.setRow(i, theta * _gamma1 * (_input - wi));
    }
    (*getConnection("input", "lattice")->getWeights()) += deltaW;

    Matrix deltaC(getGroup("lattice")->getDim(), getGroup("context")->getDim());
    Vector* ct = getGroup("context")->getOutput();

    for(int i = 0; i < getGroup("lattice")->getDim(); i++) {
        theta = calcNeighborhood(i, GAUSSIAN);
        Vector ci = getConnection("context", "lattice")->getWeights()->row(i);
        deltaC.setRow(i, theta * _gamma2 * (*ct - ci));
    }

    (*getConnection("context", "lattice")->getWeights()) += deltaC;
}

void MSOM::updateContext() {
    NeuralGroup* context = getGroup("context");
    Vector ct = Vector::Zero(context->getDim());
    Matrix unitary = Matrix::Identity(context->getDim(), context->getDim());

    Vector wIt = getConnection("input", "lattice")->getWeights()->row(_winner);
    Vector cIt = getConnection("context", "lattice")->getWeights()->row(_winner);

    ct = (1 - _beta) * wIt + _beta * cIt;
    cIt = ct;
}

double MSOM::calcDistance(int p_index) {
    Vector xi = getConnection("input", "lattice")->getWeights()->row(p_index);
    Vector ci = getConnection("context", "lattice")->getWeights()->row(p_index);
    Vector* xt = getGroup("input")->getOutput();
    Vector* ct = getGroup("context")->getOutput();

    double dt = (1 - _alpha) * pow(vectorDistance(xt, &xi), 2) + _alpha * pow(vectorDistance(ct, &ci),2);
    return dt;
}

void MSOM::initTraining(double p_gamma1, double p_gamma2, double p_alpha, double p_beta, double p_epochs) {
    _iteration = 0;
    _qError = 0;
    _alpha = p_alpha;
    _beta = p_beta;
    _gamma1_0 = p_gamma1;
    _gamma2_0 = p_gamma2;
    _lambda = p_epochs / log(_sigma0);
    _sigma =  _sigma0 * exp(-_iteration/_lambda);
    _gamma1 =  _gamma1_0 * exp(-_iteration/_lambda);
    _gamma2 =  _gamma2_0 * exp(-_iteration/_lambda);
}

void MSOM::initTraining(double p_alpha, double p_epochs) {
}

void MSOM::paramDecay() {
    _iteration++;
    _qError = 0;
    _sigma =  _sigma0 * exp(-_iteration/_lambda);
    _gamma1 =  _gamma1_0 * exp(-_iteration/_lambda);
    _gamma2 =  _gamma2_0 * exp(-_iteration/_lambda);
}

void MSOM::resetContext() {
    getConnection("context", "lattice")->getWeights()->fill(0);
}

json MSOM::getFileData() {
    return json({{"type", "msom"},{"dimx", _dimX}, {"dimy", _dimY}});
}
