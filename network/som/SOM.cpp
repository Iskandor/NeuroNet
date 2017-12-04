//
// Created by mpechac on 8. 3. 2016.
//

#include <math.h>
#include "SOM.h"
#include "../Define.h"

using namespace NeuroNet;

SOM::SOM(int p_dimInput, int p_dimX, int p_dimY, NeuralGroup::ACTIVATION p_actFunction) : NeuralNetwork() {
    addLayer("input", p_dimInput, NeuralGroup::IDENTITY, INPUT);
    addLayer("lattice", p_dimX * p_dimY, p_actFunction, OUTPUT);
    addConnection("input", "lattice");

    _winner = 0;
    _sigma0 = sqrt(max(p_dimX, p_dimY));
    _lambda = 1;
    _qError = 0;

    _dimX = p_dimX;
    _dimY = p_dimY;
}

SOM::~SOM(void) {
}

void SOM::train(Vector *p_input) {
    setInput(p_input);
    onLoop();

    findWinner();
    updateWeights();
}

void SOM::findWinner() {
    double winnerDist = INFINITY;
    double neuronDist = 0;
    _winner = 0;

    for(int i = 0; i < getGroup("lattice")->getDim(); i++) {
        neuronDist = calcDistance(i);
        if (winnerDist > neuronDist) {
            _winner = i;
            winnerDist = neuronDist;
        }
    }

    _qError += winnerDist;
    _winnerSet.insert(_winner);
}

void SOM::updateWeights() {
    int dimInput = getGroup("input")->getDim();
    int dimLattice = getGroup("lattice")->getDim();

    Matrix deltaW(dimLattice, dimInput);
    Matrix* wi = getConnection("input", "lattice")->getWeights();

    double theta = 0;

    for(int i = 0; i < dimLattice; i++) {
        theta = calcNeighborhood(i, GAUSSIAN);
        for(int j = 0; j < dimInput; j++) {
            deltaW.set(i, j, theta * _alpha * ((*_inputGroup->getOutput())[j] - wi->at(i, j)));
        }
    }

    (*getConnection("input", "lattice")->getWeights()) += deltaW;
}

void SOM::activate(Vector *p_input) {
    double neuronDist = 0;
    _winner = 0;

    setInput(p_input);
    findWinner();

    for(int i = 0; i < getGroup("lattice")->getDim(); i++) {
        neuronDist = calcDistance(i);
        switch(getOutputGroup()->getActivationFunction()) {
            case NeuralGroup::LINEAR:
                _output[i] = neuronDist;
                break;
            case NeuralGroup::EXPONENTIAL:
                _output[i] = (double) exp(-neuronDist);
                break;
            default:
                break;
        }
    }
}

double SOM::calcDistance(int p_index) {
    Vector* input = _inputGroup->getOutput();
    Matrix* weights = getConnection("input", "lattice")->getWeights();

    int dim = _inputGroup->getDim();
    double s = 0;

    for(int i = 0; i < dim; i++) {
        s+= pow((*input)[i] - weights->at(p_index, i), 2);
    }

    return (double) sqrt(s);
}

double SOM::calcNeighborhood(int p_index, NEIGHBORHOOD_TYPE p_type) {
    int x1,x2,y1,y2;
    double result = 0;

    x1 = p_index % _dimX;
    y1 = p_index / _dimX;
    x2 = _winner % _dimX;
    y2 = _winner / _dimX;

    switch (p_type) {
        case NEIGHBORHOOD_TYPE::EUCLIDEAN:
            result = 1.0 / euclideanDistance(x1, y1, x2, y2);
            break;
        case NEIGHBORHOOD_TYPE::GAUSSIAN:
            result = gaussianDistance(euclideanDistance(x1, y1, x2, y2), _sigma);
            break;
    }

    return result;
}

void SOM::initTraining(double p_alpha, double p_epochs) {
    _iteration = 0;
    _qError = 0;
    _winnerSet.clear();
    _alpha0 = _alpha = p_alpha;
    _lambda = (double) (p_epochs / log(_sigma0));
    _sigma = (double) (_sigma0 * exp(-_iteration / _lambda));
    _alpha = (double) (_alpha0 * exp(-_iteration / _lambda));
}

void SOM::paramDecay() {
    _iteration++;
    _qError = 0;
    _sigma = (double) (_sigma0 * exp(-_iteration / _lambda));
    _alpha = (double) (_alpha0 * exp(-_iteration / _lambda));
}

double SOM::euclideanDistance(int p_x1, int p_y1, int p_x2, int p_y2) {
    return sqrt(pow(p_x1 - p_x2, 2) + pow(p_y1 - p_y2, 2));
}

double SOM::gaussianDistance(double p_d, double p_sigma) {
    return (double) (exp(-pow(p_d, 2) / 2 * p_sigma) / (p_sigma * sqrt(2 * PI)));
}

double SOM::getWinnerDifferentiation() {
    return (double)_winnerSet.size()/ (double)getGroup("lattice")->getDim();
}

json SOM::getFileData() {
    return json({{"type", "som"},{"dimx", _dimX}, {"dimy", _dimY}});
}
