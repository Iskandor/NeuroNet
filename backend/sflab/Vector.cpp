//
// Created by mpechac on 15. 3. 2017.
//

#include "Vector.h"

using namespace SFLAB;


Vector::Vector(int p_dim, const Base::INIT &p_init, double p_value) : Base(p_dim, 1) {
    init(p_init, p_value);
}

Vector::Vector(int p_rows, int p_cols, const INIT &p_init, double p_value) : Base(p_rows, p_cols) {
    init(p_init, p_value);
}

Vector::Vector(const Vector &p_copy) : Base(p_copy) {
}

Vector::~Vector() {
}

void Vector::operator=(const Vector &p_vector) {
    Base::clone(p_vector);
}

void Vector::init(INIT p_init, double p_value) {
    switch(p_init) {
        case ZERO:
            fill(0);
            break;
        case ONES:
            fill(1);
            break;
        case VALUE:
            fill(p_value);
            break;
    }
}

Vector Vector::T() {
    Vector res(0);

    if (_cols == 1) {
        res._rows = 1;
        res._cols = _rows;
        res.internal_init();

        for(int i = 0; i < _rows; i++) {
            res._arr[0][i] = _arr[i][0];
        }
    }

    if (_rows == 1) {
        res._rows = _cols;
        res._cols = 1;
        res.internal_init();

        for(int i = 0; i < _cols; i++) {
            res._arr[i][0] = _arr[0][i];
        }
    }

    return Vector(res);
}

Vector Vector::operator+(const Vector &p_vector) {
    if (_cols == 1) {
        Vector res(_rows);

        for(int i = 0; i < _rows; i++) {
            res._arr[i][0] = _arr[i][0] + p_vector._arr[i][0];
        }

        return Vector(res);
    }

    if (_rows == 1) {
        Vector res(_cols);

        for(int i = 0; i < _cols; i++) {
            res._arr[0][i] = _arr[0][i] + p_vector._arr[0][i];
        }

        return Vector(res);
    }

    return Vector(0);
}

Vector Vector::operator-(const Vector &p_vector) {
    if (_cols == 1) {
        Vector res(_rows);

        for (int i = 0; i < _rows; i++) {
            res._arr[i][0] = _arr[i][0] + p_vector._arr[i][0];
        }

        return Vector(res);
    }

    if (_rows == 1) {
        Vector res(_cols);

        for (int i = 0; i < _cols; i++) {
            res._arr[0][i] = _arr[0][i] + p_vector._arr[0][i];
        }

        return Vector(res);
    }

    return Vector(0);
}

Matrix Vector::operator*(const Vector &p_vector) {
    Matrix res(_rows, p_vector._cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < p_vector._cols; j++) {
            res[i][j] = _arr[i][0] * p_vector._arr[0][j];
        }
    }

    return Matrix(res);
}

Vector Vector::operator*(const double p_const) {
    if (_cols == 1) {
        Vector res(_rows);

        for (int i = 0; i < _rows; i++) {
            res._arr[i][0] = _arr[i][0] * p_const;
        }

        return Vector(res);
    }

    if (_rows == 1) {
        Vector res(_cols);

        for (int i = 0; i < _cols; i++) {
            res._arr[0][i] = _arr[0][i] * p_const;
        }

        return Vector(res);
    }

    return Vector(0);
}

double &Vector::operator[](int p_index) {
    double* res = nullptr;
    if (_cols == 1) {
        res = &_arr[p_index][0];
    }

    if (_rows == 1) {
        res = &_arr[0][p_index];
    }

    return *res;
}
