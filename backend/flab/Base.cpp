//
// Created by mpechac on 15. 3. 2017.
//

#include <stdlib.h>
#include <malloc.h>
#include "Base.h"

using namespace FLAB;

Base::Base(int p_rows, int p_cols) {
    _rows = p_rows;
    _cols = p_cols;

    if (_rows != 0 && _cols != 0) {
        internal_init();
    }
}

Base::Base(int p_rows, int p_cols, double *p_data) {
    _rows = p_rows;
    _cols = p_cols;

    if (_rows != 0 && _cols != 0) {
        internal_init(p_data);
    }
}

Base::Base(int p_rows, int p_cols, initializer_list<double> p_inputs) {
    _rows = p_rows;
    _cols = p_cols;

    if (_rows != 0 && _cols != 0) {
        internal_init(p_inputs);
    }
}

Base::Base(const Base &p_copy) {
    clone(p_copy);
}

Base::~Base() {
    if (_arr != NULL) {
        free(_arr);
    }
}

void Base::fill(double p_value) {
    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            _arr[i * _cols + j] = p_value;
        }
    }
}

void Base::clone(const Base &p_copy) {
    if (_arr != NULL) {
        free(_arr);
    }

    _rows = p_copy._rows;
    _cols = p_copy._cols;
    _arr = Base::allocBuffer(_rows, _cols);
    memcpy(_arr, p_copy._arr, sizeof(double) * (size_t) (_rows * _cols));
}

void Base::internal_init(double *p_data) {
    if (p_data != NULL) {
        _arr = p_data;
    }
    else {
        _arr = Base::allocBuffer(_rows, _cols);
    }
}

void Base::internal_init(initializer_list<double> p_inputs) {
    _arr = Base::allocBuffer(_rows, _cols);

    int i = 0;
    int j = 0;

    for(double in: p_inputs) {
        _arr[i * _cols + j] = in;
        j++;
        if (j == _cols) {
            i++;
            j = 0;
        }
    }
}

double* Base::allocBuffer(int p_rows, int p_cols) {
    return (double*)calloc((size_t) (p_rows * p_cols), sizeof(double));
}

double Base::maxCoeff() {
    double res = _arr[0];

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; i++) {
            if (res < _arr[i * _cols + j]) {
                res = _arr[i * _cols + j];
            };
        }
    }

    return res;
}

double Base::minCoeff() {
    double res = _arr[0];

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; i++) {
            if (res > _arr[i * _cols + j]) {
                res = _arr[i * _cols + j];
            };
        }
    }

    return res;
}
