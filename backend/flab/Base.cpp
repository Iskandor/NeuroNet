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

Base::Base(int p_rows, int p_cols, double **p_data) {
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
        for(int i = 0; i < _rows; i++) {
            free(_arr[i]);
        }
        free(_arr);
    }
}

void Base::fill(double p_value) {
    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            _arr[i][j] = p_value;
        }
    }
}

void Base::init(INIT p_init, double p_value) {

}

void Base::clone(const Base &p_copy) {
    if (_arr != NULL) {
        for(int i = 0; i < _rows; i++) {
            if (_arr[i] != NULL) free(_arr[i]);
        }
        free(_arr);
    }

    _rows = p_copy._rows;
    _cols = p_copy._cols;
    _arr = (double**)calloc((size_t) (_rows), sizeof(double*));

    for(int i = 0; i < _rows; i++) {
        _arr[i] = (double*)calloc((size_t) (_cols), sizeof(double));
        memcpy(_arr[i], p_copy._arr[i], sizeof(double) * (size_t) (_cols));
    }
}

void Base::internal_init(double *p_data) {
    _arr = (double **) calloc((size_t) (_rows), sizeof(double *));

    for (int i = 0; i < _rows; i++) {
        _arr[i] = (double *) calloc((size_t) (_cols), sizeof(double));
    }

    if (p_data != NULL) {
        for (int i = 0; i < _rows; i++) {
            for (int j = 0; j < _cols; j++) {
                _arr[i][j] = p_data[i * _cols + j];
            }
        }
    }
}

void Base::internal_init(double **p_data) {
    _arr = p_data;
}

void Base::internal_init(initializer_list<double> p_inputs) {
    _arr = (double**)calloc((size_t) (_rows), sizeof(double*));

    for(int i = 0; i < _rows; i++) {
        _arr[i] = (double*)calloc((size_t) (_cols), sizeof(double));
    }

    int i = 0;
    int j = 0;

    for(double in: p_inputs) {
        _arr[i][j] = in;
        j++;
        if (j == _cols) {
            i++;
            j = 0;
        }
    }

}

double** Base::allocBuffer(int p_rows, int p_cols) {
    double **p_data = (double**)calloc((size_t) (p_rows), sizeof(double*));

    for(int i = 0; i < p_rows; i++) {
        p_data[i] = (double*)calloc((size_t) (p_cols), sizeof(double));
    }

    return p_data;
}

double Base::maxCoeff() {
    double res = _arr[0][0];

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; i++) {
            if (res < _arr[i][j]) {
                res = _arr[i][j];
            };
        }
    }

    return res;
}

double Base::minCoeff() {
    double res = _arr[0][0];

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; i++) {
            if (res > _arr[i][j]) {
                res = _arr[i][j];
            };
        }
    }

    return res;
}
