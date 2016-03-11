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
    _error = 0;

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
}

void SOM::updateWeights() {
    MatrixXd delta(getGroup("lattice")->getDim(), getGroup("input")->getDim());
    double theta = 0;

    for(int i = 0; i < getGroup("lattice")->getDim(); i++) {
        theta = calcNeighborhood(i, EUCLIDEAN);
        for(int j = 0; j < getGroup("input")->getDim(); j++) {
            delta(i, j) = theta * _alpha * (_input[j] - (*getConnection("input", "lattice")->getWeights())(i, j));
        }
    }

    (*getConnection("input", "lattice")->getWeights()) += delta;
}

double SOM::calcDistance(int p_index) {
    double result = 0;
    for(int i = 0; i < getGroup("input")->getDim(); i++) {
        result += pow(_input[i] - (*getConnection("input", "lattice")->getWeights())(p_index, i), 2);
    }

    _error += sqrt(result);

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
            result = euclideanDistance(x1, y1, x2, y2);
            break;
        case GAUSSIAN:
            result = gaussianDistance(euclideanDistance(x1, y1, x2, y2), _sigma);
            break;
    }

    return result;
}

void SOM::reset(double p_alpha) {
    _iteration = 0;
    _alpha0 = _alpha = p_alpha;
}

void SOM::paramDecay() {
    _iteration++;
    _lambda = _iteration / log(_sigma0);
    _sigma =  _sigma0 * exp(-_iteration/_lambda);
    _alpha =  _alpha0 * exp(-_iteration/_lambda);
    _error = 0;
}

double SOM::euclideanDistance(int p_x1, int p_y1, int p_x2, int p_y2) {
    return sqrt(pow(p_x1 - p_x2, 2) + pow(p_y1 - p_y2, 2));
}

double SOM::gaussianDistance(int p_d, double p_sigma) {
    return exp(-pow(p_d,2) / 2 * p_sigma) / (p_sigma * sqrt(2*PI));
}
