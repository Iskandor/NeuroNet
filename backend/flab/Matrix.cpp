//
// Created by mpechac on 10. 3. 2017.
//

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Matrix.h"
#include "RandomGenerator.h"

using namespace FLAB;

Matrix::Matrix(int p_rows, int p_cols, INIT p_init, double p_value) : Base(p_rows, p_cols) {
    init(p_init, p_value);
}

Matrix::Matrix(int p_rows, int p_cols, double *p_data) : Base(p_rows, p_cols, p_data) {
}

Matrix::Matrix(int p_rows, int p_cols, std::initializer_list<double> inputs) : Base(p_rows, p_cols, inputs) {
}

Matrix::Matrix(const Matrix &p_copy) : Base(p_copy) {

}

void Matrix::init(INIT p_init, double p_value) {
    switch(p_init) {
        case ZERO:
            fill(0);
            break;
        case IDENTITY:
            fill(0);
            for(int i = 0; i < _rows; i++) {
                _arr[i * _cols + i] = 1;
            }
            break;
        case VALUE:
            fill(p_value);
            break;
        case RANDOM:
            for(int i = 0; i < _rows; i++) {
                for(int j = 0; j < _cols; j++) {
                    _arr[i * _cols + j] = RandomGenerator::getInstance().random(-1, 1);
                }
            }
            break;

    }
}

void Matrix::operator=(const Matrix &p_matrix) {
    Base::clone(p_matrix);
}

Matrix Matrix::operator+(const Matrix &p_matrix) {
    double* data = Base::allocBuffer(_rows, _cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            data[i * _cols + j] = _arr[i * _cols + j] + p_matrix._arr[i * _cols + j];
        }
    }

    return Matrix(_rows, _cols, data);
}
void Matrix::operator+=(const Matrix &p_matrix) {
    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            _arr[i * _cols + j] += p_matrix._arr[i * _cols + j];
        }
    }
}

Matrix Matrix::operator-(const Matrix &p_matrix) {
    double* data = Base::allocBuffer(_rows, _cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            data[i * _cols + j] = _arr[i * _cols + j] - p_matrix._arr[i * _cols + j];
        }
    }

    return Matrix(_rows, _cols, data);
}

void Matrix::operator-=(const Matrix &p_matrix) {
    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            _arr[i * _cols + j] -= _arr[i * _cols + j] - p_matrix._arr[i * _cols + j];
        }
    }
}

Matrix Matrix::operator*(const Matrix &p_matrix) {
    double* data = Base::allocBuffer(_rows, p_matrix._cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < p_matrix._cols; j++) {
            for(int k = 0; k < _cols; k++) {
                data[i * _cols + j] += _arr[i * _cols + k] * p_matrix._arr[k * p_matrix._cols + j];
            }
        }
    }

    return Matrix(_rows, p_matrix._cols, data);
}

Vector Matrix::operator*(Vector p_vector) {
    double* data = Base::allocBuffer(_rows, 1);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            data[i] = data[i] + _arr[i * _cols + j] * p_vector[j];
        }
    }

    return Vector(_rows, 1, data);
}

Matrix Matrix::operator*(const double p_const) {
    double* data = Base::allocBuffer(_rows, _cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            data[i * _cols + j] = _arr[i * _cols + j] * p_const;
        }
    }

    return Matrix(_rows, _cols, data);
}

void Matrix::operator*=(const double p_const) {
    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            _arr[i * _cols + j] *= p_const;
        }
    }
}

Matrix Matrix::T() {
    double* data = Base::allocBuffer(_cols, _rows);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            data[j * _rows + i] = _arr[i * _cols + j];
        }
    }

    return Matrix(_cols, _rows, data);
}

Matrix Matrix::inv() {
    double* data = Base::allocBuffer(_rows, _cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            data[i * _cols + j] = 1.0 / _arr[i * _cols + j];
        }
    }

    return Matrix(_rows, _cols, data);
}

Matrix Matrix::Zero(int p_rows, int p_cols) {
    return Matrix(p_rows, p_cols, ZERO);
}

Matrix Matrix::Random(int p_rows, int p_cols) {
    return Matrix(p_rows, p_cols, RANDOM);
}

Matrix Matrix::Identity(int p_rows, int p_cols) {
    return Matrix(p_rows, p_cols, IDENTITY);
}

Matrix Matrix::Value(int p_rows, int p_cols, double p_value) {
    return Matrix(p_rows, p_cols, VALUE, p_value);
}

Matrix Matrix::ew_sqrt() {
    double* data = Base::allocBuffer(_rows, _cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            data[i * _cols + j] = (double) sqrt(_arr[i * _cols + j]);
        }
    }

    return Matrix(_rows, _cols, data);
}

Matrix Matrix::ew_pow(int p_n) {
    double* data = Base::allocBuffer(_rows, _cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            data[i * _cols + j] = (double) pow(_arr[i * _cols + j], p_n);
        }
    }

    return Matrix(_rows, _cols, data);
}

Matrix Matrix::ew_dot(const Matrix &p_matrix) {
    double* data = Base::allocBuffer(_rows, _cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            data[i * _cols + j] = _arr[i * _cols + j] * p_matrix._arr[i * p_matrix._cols + j];
        }
    }

    return Matrix(_rows, _cols, data);
}
