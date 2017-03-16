//
// Created by mpechac on 10. 3. 2017.
//

#include <stdlib.h>
#include <string.h>
#include "Matrix.h"

using namespace SFLAB;

Matrix::Matrix(int p_rows, int p_cols, INIT p_init, double p_value) : Base(p_rows, p_cols) {
    init(p_init, p_value);
}

Matrix::Matrix(const Matrix &p_copy) : Base(p_copy) {

}

void Matrix::init(INIT p_init, double p_value) {
    switch(p_init) {
        case ZERO:
            fill(0);
            break;
        case UNITARY:
            fill(0);
            for(int i = 0; i < _rows; i++) {
                _arr[i][i] = 1;
            }
            break;
        case VALUE:
            fill(p_value);
            break;
    }
}

void Matrix::operator=(const Matrix &p_matrix) {
    Base::clone(p_matrix);
}

Matrix Matrix::operator+(const Matrix &p_matrix) {
    Matrix res(_rows, _cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            res._arr[i][j] = _arr[i][j] + p_matrix._arr[i][j];
        }
    }

    return res;
}

Matrix Matrix::operator-(const Matrix &p_matrix) {
    Matrix res(_rows, _cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            res._arr[i][j] = _arr[i][j] - p_matrix._arr[i][j];
        }
    }

    return Matrix(res);
}


Matrix Matrix::operator*(const Matrix &p_matrix) {
    Matrix res(_rows, p_matrix._cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < p_matrix._cols; j++) {
            for(int k = 0; k < _cols; k++) {
                res._arr[j][i] += _arr[i][k] * p_matrix._arr[k][j];
            }
        }
    }

    return Matrix(res);
}

Vector Matrix::operator*(Vector p_vector) {
    Vector res(_rows);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            res[i] = res[i] + _arr[i][j] * p_vector[j];
        }
    }

    return Vector(res);
}

Matrix Matrix::operator*(const double p_const) {
    Matrix res(_cols, _rows);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            res._arr[i][j] *= p_const;
        }
    }

    return Matrix(res);
}

Matrix Matrix::T() {
    Matrix res(_cols, _rows);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            res._arr[j][i] = _arr[i][j];
        }
    }

    return Matrix(res);
}

Matrix Matrix::inv() {
    Matrix res(_rows, _cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            res._arr[i][j] = 1.0 / _arr[i][j];
        }
    }

    return Matrix(res);
}
