//
// Created by mpechac on 28. 3. 2017.
//

#include <assert.h>
#include <malloc.h>
#include "Tensor3.h"
#include "RandomGenerator.h"

using namespace FLAB;

Tensor3::Tensor3(int p_x, int p_y, int p_z, INIT p_init, double p_value) {
    _dims.push_back(p_x);
    _dims.push_back(p_y);
    _dims.push_back(p_z);
    _dim = calcDim();

    if (_dim > 0) {
        _arr = (double *) calloc((size_t) (_dim), sizeof(double));
        init(p_init, p_value);
    }
}

Tensor3::Tensor3(int p_x, int p_y, int p_z, double *p_data) {
    _dims.push_back(p_x);
    _dims.push_back(p_y);
    _dims.push_back(p_z);
    _dim = calcDim();
    _arr = (double *) calloc((size_t) (_dim), sizeof(double));

    if (p_data != NULL) {
        for(int i = 0; i < _dim; i++) {
            _arr[i] = p_data[i];
        }
    }
}

Tensor3::Tensor3(const Tensor3 &p_copy) {
    clone(p_copy);
}

void Tensor3::clone(const Tensor3 &p_copy) {
    _dims.clear();
    _dims = p_copy._dims;
    _dim = calcDim();
    _arr = (double *) calloc((size_t) (_dim), sizeof(double));

    memcpy(_arr, p_copy._arr, sizeof(double) * (size_t) (_dim));
}

Tensor3::~Tensor3() {
    if (_arr != NULL) {
        free(_arr);
    }
}

void Tensor3::fill(double p_value) {
    for(int i = 0; i < _dim; i++) {
        _arr[i] = p_value;
    }
}

void Tensor3::init(Tensor3::INIT p_init, double p_value) {
    switch (p_init) {
        case ZERO:
            fill(0);
            break;
        case ONES:
            fill(1);
            break;
        case VALUE:
            fill(p_value);
            break;
        case RANDOM:
            for(int i = 0; i < _dim; i++) {
                _arr[i] = RandomGenerator::getInstance().random(-1, 1);
            }
            break;
    }
}

double Tensor3::operator()(int x, int y, int z) {
    int index = x + y * _dims[0] + z * _dims[0] * _dims[1];

    if (index >= _dim) {
        assert(index < _dim);
    }

    return _arr[index];
}

void Tensor3::set(int x, int y, int z, double p_value) {
    int index = x + y * _dims[0] + z * _dims[0] * _dims[1];

    if (index >= _dim) {
        assert(index < _dim);
    }

    _arr[index] = p_value;
}

int Tensor3::calcDim() {
    int dim = 1;
    for(int i = 0; i < _dims.size(); i++) {
        dim *= _dims[i];
    }

    return dim;
}

Tensor3 Tensor3::Zero(int p_x, int p_y, int p_z) {
    return Tensor3(p_x, p_y, p_z, ZERO);
}

Tensor3 Tensor3::Random(int p_x, int p_y, int p_z) {
    return Tensor3(p_x, p_y, p_z, RANDOM);
}

Tensor3 Tensor3::Ones(int p_x, int p_y, int p_z) {
    return Tensor3(p_x, p_y, p_z, ONES);
}

Tensor3 Tensor3::Value(int p_x, int p_y, int p_z, double p_value) {
    return Tensor3(p_x, p_y, p_z, VALUE, p_value);
}
