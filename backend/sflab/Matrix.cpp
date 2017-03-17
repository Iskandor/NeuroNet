//
// Created by mpechac on 10. 3. 2017.
//

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Matrix.h"
#include "RandomGenerator.h"

using namespace SFLAB;

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
                _arr[i][i] = 1;
            }
            break;
        case VALUE:
            fill(p_value);
            break;
        case RANDOM:
            for(int i = 0; i < _rows; i++) {
                for(int j = 0; j < _cols; j++) {
                    _arr[i][j] = RandomGenerator::getInstance().random(-1, 1);
                }
            }
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
void Matrix::operator+=(const Matrix &p_matrix) {
    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            _arr[i][j] += p_matrix._arr[i][j];
        }
    }
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

void Matrix::operator-=(const Matrix &p_matrix) {
    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            _arr[i][j] -= _arr[i][j] - p_matrix._arr[i][j];
        }
    }
}

Matrix Matrix::operator*(const Matrix &p_matrix) {
    Matrix res(_rows, p_matrix._cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < p_matrix._cols; j++) {
            for(int k = 0; k < _cols; k++) {
                res._arr[i][j] += _arr[i][k] * p_matrix._arr[k][j];
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

void Matrix::operator*=(const double p_const) {
    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            _arr[i][j] *= p_const;
        }
    }
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

Matrix Matrix::Zero(int p_rows, int p_cols) {
    Matrix res(p_rows, p_cols, ZERO);
    return Matrix(res);
}

Matrix Matrix::Random(int p_rows, int p_cols) {
    Matrix res(p_rows, p_cols, RANDOM);

    return Matrix(res);
}

Matrix Matrix::Identity(int p_rows, int p_cols) {
    Matrix res(p_rows, p_cols, IDENTITY);

    return Matrix(res);
}

Vector Matrix::row(int p_index) {
    return Vector(_cols, _arr[p_index]);;
}

Vector Matrix::col(int p_index) {
    double data[_rows];

    for (int i = 0; i < _rows; i++) {
        data[i] = _arr[i][p_index];
    }

    return Vector(_rows, data);
}

Matrix Matrix::Value(int p_rows, int p_cols, double p_value) {
    return Matrix(p_rows, p_cols, VALUE, p_value);
}

Matrix Matrix::ew_sqrt() {
    Matrix res(_rows, _cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            res._arr[i][j] = sqrt(_arr[i][j]);
        }
    }

    return Matrix(res);
}

Matrix Matrix::ew_pow(int p_n) {
    Matrix res(_rows, _cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            res._arr[i][j] = pow(_arr[i][j], p_n);
        }
    }

    return Matrix(res);
}

Matrix Matrix::ew_dot(const Matrix &p_matrix) {
    Matrix res(_rows, _cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < _cols; j++) {
            res._arr[i][j] = _arr[i][j] * p_matrix._arr[i][j];
        }
    }

    return Matrix(res);
}
