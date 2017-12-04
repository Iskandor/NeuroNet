//
// Created by mpechac on 19. 5. 2016.
//

#include "RecSOM.h"
#include "../Define.h"

using namespace NeuroNet;

RecSOM::RecSOM(int p_dimInput, int p_dimX, int p_dimY, NeuralGroup::ACTIVATION p_actFunction) : SOM(p_dimInput, p_dimX, p_dimY, p_actFunction) {
    addLayer("context", p_dimX * p_dimY, NeuralGroup::IDENTITY, NeuralNetwork::HIDDEN);
    addConnection("context", "lattice");
}

RecSOM::~RecSOM() {

}

void RecSOM::train(Vector *p_input) {
    setInput(p_input);

    findWinner();
    updateWeights();
    updateContext();
}

void RecSOM::activate(Vector *p_input) {
    SOM::activate(p_input);
}

void RecSOM::updateWeights() {
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
}

void RecSOM::updateContext() {
    NeuralGroup* context = getGroup("context");
    Vector ct = Vector::Zero(context->getDim());
    Matrix unitary = Matrix::Identity(context->getDim(), context->getDim());

    double neuronDist = 0;

    for(int i = 0; i < getGroup("lattice")->getDim(); i++) {
        neuronDist = calcDistance(i);
        switch(getOutputGroup()->getActivationFunction()) {
            case NeuralGroup::LINEAR:
                ct[i] = neuronDist;
                break;
            case NeuralGroup::EXPONENTIAL:
                ct[i] = exp(-neuronDist);
                break;
        }
    }

    context->integrate(&ct, &unitary);
    context->fire();
}

double RecSOM::calcDistance(int p_index) {
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

    double dt = _alpha * dx + _beta * dc;
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
