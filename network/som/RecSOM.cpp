//
// Created by mpechac on 19. 5. 2016.
//

#include "RecSOM.h"
#include "../Define.h"

using namespace NeuroNet;

RecSOM::RecSOM(int p_dimInput, int p_dimX, int p_dimY, NeuralGroup::ACTIVATION_FN p_actFunction) : SOM(p_dimInput, p_dimX, p_dimY, p_actFunction) {
    addLayer("context", p_dimX * p_dimY, NeuralGroup::IDENTITY, NeuralNetwork::HIDDEN);
    addConnection("context", "lattice");
}

RecSOM::~RecSOM() {

}

void RecSOM::train(double *p_input) {
    setInput(p_input);
    onLoop();

    findWinner();
    updateWeights();
    updateContext();
}

void RecSOM::activate(VectorXd *p_input) {
    SOM::activate(p_input);
}

void RecSOM::updateWeights() {
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

void RecSOM::updateContext() {
    NeuralGroup* context = getGroup("context");
    VectorXd ct = VectorXd::Zero(context->getDim());
    MatrixXd unitary = MatrixXd::Identity(context->getDim(), context->getDim());

    double neuronDist = 0;

    for(int i = 0; i < getGroup("lattice")->getDim(); i++) {
        neuronDist = calcDistance(i);
        switch(getOutputGroup()->getActivationFunction()) {
            case LINEAR:
                ct[i] = neuronDist;
                break;
            case EXPONENTIAL:
                ct[i] = exp(-neuronDist);
                break;
        }
    }

    context->integrate(&ct, &unitary);
    context->fire();
}

double RecSOM::calcDistance(int p_index) {
    VectorXd xi = getConnection("input", "lattice")->getWeights()->row(p_index);
    VectorXd ci = getConnection("context", "lattice")->getWeights()->row(p_index);
    VectorXd* xt = getGroup("input")->getOutput();
    VectorXd* ct = getGroup("context")->getOutput();

    double dt = _alpha * pow(vectorDistance(xt, &xi), 2) + _beta * pow(vectorDistance(ct, &ci),2);
    return dt;
}

void RecSOM::initTraining(double p_gamma1, double p_gamma2, double p_alpha, double p_beta, double p_epochs) {
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

void RecSOM::initTraining(double p_alpha, double p_epochs) {
}

void RecSOM::paramDecay() {
    _iteration++;
    _qError = 0;
    _sigma =  _sigma0 * exp(-_iteration/_lambda);
    _gamma1 =  _gamma1_0 * exp(-_iteration/_lambda);
    _gamma2 =  _gamma2_0 * exp(-_iteration/_lambda);
}

json RecSOM::getFileData() {
    return json({{"type", "recsom"},{"dimx", _dimX}, {"dimy", _dimY}});
}
