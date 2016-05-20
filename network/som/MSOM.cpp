//
// Created by mpechac on 13. 4. 2016.
//

#include "MSOM.h"
#include "../Define.h"

using namespace NeuroNet;

MSOM::MSOM(int p_dimInput, int p_dimX, int p_dimY, int p_actFunction) : SOM(p_dimInput, p_dimX, p_dimY, p_actFunction) {
    addLayer("context", p_dimInput, IDENTITY, NeuralNetwork::HIDDEN);
    addConnection("context", "lattice");
}

MSOM::~MSOM() {

}

void MSOM::train(double *p_input) {
    setInput(p_input);
    onLoop();

    findWinner();
    updateWeights();
    updateContext();
}

void MSOM::activate(VectorXd *p_input) {
    SOM::activate(p_input);
}

void MSOM::updateWeights() {
    MatrixXd deltaW(getGroup("lattice")->getDim(), getGroup("input")->getDim());
    double theta = 0;

    for(int i = 0; i < getGroup("lattice")->getDim(); i++) {
        theta = calcNeighborhood(i, GAUSSIAN);
        VectorXd wi = getConnection("input", "lattice")->getWeights()->row(i);
        deltaW.row(i) = theta * _gamma1 * (_input - wi);
    }
    (*getConnection("input", "lattice")->getWeights()) += deltaW;

    MatrixXd deltaC(getGroup("lattice")->getDim(), getGroup("context")->getDim());
    VectorXd* ct = getGroup("context")->getOutput();

    for(int i = 0; i < getGroup("lattice")->getDim(); i++) {
        theta = calcNeighborhood(i, GAUSSIAN);
        VectorXd ci = getConnection("context", "lattice")->getWeights()->row(i);
        deltaC.row(i) = theta * _gamma2 * (*ct - ci);
    }

    (*getConnection("context", "lattice")->getWeights()) += deltaC;
}

void MSOM::updateContext() {
    NeuralGroup* context = getGroup("context");
    VectorXd ct = VectorXd::Zero(context->getDim());
    MatrixXd unitary = MatrixXd::Identity(context->getDim(), context->getDim());

    VectorXd wIt = getConnection("input", "lattice")->getWeights()->row(_winner);
    VectorXd cIt = getConnection("context", "lattice")->getWeights()->row(_winner);

    ct = (1 - _beta) * wIt + _beta * cIt;
    cIt = ct;
}

double MSOM::calcDistance(int p_index) {
    VectorXd xi = getConnection("input", "lattice")->getWeights()->row(p_index);
    VectorXd ci = getConnection("context", "lattice")->getWeights()->row(p_index);
    VectorXd* xt = getGroup("input")->getOutput();
    VectorXd* ct = getGroup("context")->getOutput();

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
