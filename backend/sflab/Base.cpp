//
// Created by mpechac on 15. 3. 2017.
//

#include "Base.h"

using namespace SFLAB;

Base::Base(int p_rows, int p_cols) {
    _rows = p_rows;
    _cols = p_cols;

    if (_rows != 0 && _cols != 0) {
        internal_init();
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

void Base::internal_init() {
    _arr = (double**)calloc((size_t) (_rows), sizeof(double*));

    for(int i = 0; i < _rows; i++) {
        _arr[i] = (double*)calloc((size_t) (_cols), sizeof(double));
    }
}
