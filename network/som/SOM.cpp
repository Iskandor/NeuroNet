//
// Created by mpechac on 8. 3. 2016.
//

#include "SOM.h"
#include "../Define.h"

SOM::SOM(int p_dimInput, int p_dimX, int p_dimY, int p_actFunction) : NeuralNetwork() {
    addLayer("input", p_dimInput, IDENTITY, INPUT);
    addLayer("lattice", p_dimX * p_dimY, p_actFunction, OUTPUT);
    addConnection("input", "lattice");

    _sigma0 = max(p_dimX, p_dimY) / 2;
    _lambda = 1;
    _qError = 0;

    _dimX = p_dimX;
    _dimY = p_dimY;
}

SOM::~SOM(void) {
}

void SOM::train(double *p_input) {
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
    MatrixXd delta(getGroup("lattice")->getDim(), getGroup("input")->getDim());
    double theta = 0;

    for(int i = 0; i < getGroup("lattice")->getDim(); i++) {
        theta = calcNeighborhood(i, GAUSSIAN);
        for(int j = 0; j < getGroup("input")->getDim(); j++) {
            delta(i, j) = theta * _alpha * (_input[j] - (*getConnection("input", "lattice")->getWeights())(i, j));
        }
    }

    (*getConnection("input", "lattice")->getWeights()) += delta;
}

void SOM::activate(VectorXd *p_input) {
    double neuronDist = 0;
    _winner = 0;

    setInput(p_input);
    onLoop();

    for(int i = 0; i < getGroup("lattice")->getDim(); i++) {
        neuronDist = calcDistance(i);
        switch(getOutputGroup()->getActivationFunction()) {
            case LINEAR:
                _output[i] = neuronDist;
                break;
            case EXPONENTIAL:
                _output[i] = exp(-neuronDist);
                break;
        }
    }
}

double SOM::calcDistance(int p_index) {
    double result = 0;
    for(int i = 0; i < getGroup("input")->getDim(); i++) {
        result += pow(_input[i] - (*getConnection("input", "lattice")->getWeights())(p_index, i), 2);
    }

    return sqrt(result);
}

double SOM::calcNeighborhood(int p_index, NEIGHBORHOOD_TYPE p_type) {
    int x1,x2,y1,y2;
    double result = 0;

    x1 = p_index % _dimX;
    y1 = p_index / _dimX;
    x2 = _winner % _dimX;
    y2 = _winner / _dimX;

    switch (p_type) {
        case EUCLIDEAN:
            result = 1 / euclideanDistance(x1, y1, x2, y2);
            break;
        case GAUSSIAN:
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
    _lambda = p_epochs / log(_sigma0);
    _sigma =  _sigma0 * exp(-_iteration/_lambda);
    _alpha =  _alpha0 * exp(-_iteration/_lambda);
}

void SOM::paramDecay() {
    _iteration++;
    _qError = 0;
    _sigma =  _sigma0 * exp(-_iteration/_lambda);
    _alpha =  _alpha0 * exp(-_iteration/_lambda);
}

double SOM::euclideanDistance(int p_x1, int p_y1, int p_x2, int p_y2) {
    return sqrt(pow(p_x1 - p_x2, 2) + pow(p_y1 - p_y2, 2));
}

double SOM::gaussianDistance(int p_d, double p_sigma) {
    return exp(-pow(p_d,2) / 2 * p_sigma) / (p_sigma * sqrt(2*PI));
}

double SOM::getWinnerDifferentiation() {
    return (double)_winnerSet.size()/ (double)getGroup("lattice")->getDim();
}
