//
// Created by mpechac on 15. 3. 2017.
//

#include "Vector.h"
#include "RandomGenerator.h"
#include <random>

using namespace FLAB;


Vector::Vector(int p_dim, double *p_data) : Base(p_dim, 1, p_data) {
}

Vector::Vector(int p_dim, initializer_list<double> inputs) : Base(p_dim, 1, inputs) {

}

Vector::Vector(int p_dim, const Base::INIT &p_init, double p_value) : Base(p_dim, 1) {
    init(p_init, p_value);
}

Vector::Vector(int p_rows, int p_cols, const INIT &p_init, double p_value) : Base(p_rows, p_cols) {
    init(p_init, p_value);
}

Vector::Vector(int p_rows, int p_cols, double *p_data) : Base(p_rows, p_cols, p_data) {
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
        case RANDOM:
            for(int i = 0; i < _rows; i++) {
                _arr[i] = RandomGenerator::getInstance().random(-1, 1);
            }
            break;
    }
}

Vector Vector::T() {
    double* data = Base::allocBuffer(_cols, _rows);

    for(int i = 0; i < size(); i++) {
        data[i] = _arr[i];
    }

    return Vector(_cols, _rows, data);
}

Vector Vector::operator+(const Vector &p_vector) {
    double* data = Base::allocBuffer(_rows, _cols);

    for(int i = 0; i < size(); i++) {
        data[i] = _arr[i] + p_vector._arr[i];
    }

    return Vector(_rows, _cols, data);
}

Vector Vector::operator-(const Vector &p_vector) {
    double* data = Base::allocBuffer(_rows, _cols);

    for(int i = 0; i < size(); i++) {
        data[i] = _arr[i] - p_vector._arr[i];
    }

    return Vector(_rows, _cols, data);
}

Matrix Vector::operator*(const Vector &p_vector) {
    double* data = Base::allocBuffer(_rows, p_vector._cols);

    for(int i = 0; i < _rows; i++) {
        for(int j = 0; j < p_vector._cols; j++) {
            data[i * p_vector._cols + j] = _arr[i] * p_vector._arr[j];
        }
    }

    return Matrix(_rows, p_vector._cols, data);
}

Vector Vector::operator*(const double p_const) {
    double* data = Base::allocBuffer(_rows, _cols);

    for(int i = 0; i < size(); i++) {
        data[i] = _arr[i] * p_const;
    }

    return Vector(_rows, _cols, data);
}

double &Vector::operator[](int p_index) {
    return _arr[p_index];
}

void Vector::operator+=(const Vector &p_vector) {
    for(int i = 0; i < size(); i++) {
        _arr[i] = _arr[i] + p_vector._arr[i];
    }
}

void Vector::operator-=(const Vector &p_vector) {
    for(int i = 0; i < size(); i++) {
        _arr[i] = _arr[i] - p_vector._arr[i];
    }
}

void Vector::operator*=(const double p_const) {
    for(int i = 0; i < size(); i++) {
        _arr[i] *= p_const;
    }
}

double Vector::norm() {
    double res = 0;

    for(int i = 0; i < size(); i++) {
        res += pow(_arr[i], 2);
    }

    return (double) sqrt(res);
}

Vector Vector::Zero(int p_dim) {
    return Vector(p_dim);
}

Vector Vector::Random(int p_dim) {
    return Vector(p_dim, RANDOM);
}

Vector Vector::One(int p_dim) {
    return Vector(p_dim, ONES);
}

Vector Vector::Concat(Vector& p_vector1, Vector& p_vector2) {
    Vector res(p_vector1.size() + p_vector2.size());

    int index = 0;

    for(int i = 0; i < p_vector1.size(); i++) {
        res[index] = p_vector1[i];
        index++;
    }

    for(int i = 0; i < p_vector2.size(); i++) {
        res[index] = p_vector2[i];
        index++;
    }

    return Vector(res);
}

int Vector::minIndex() {
    int min = 0;

    for(int i = 0; i < size(); i++) {
        if (this->operator[](min) > this->operator[](i)) {
            min = i;
        }
    }

    return min;
}

int Vector::maxIndex() {
    int max = 0;

    for(int i = 0; i < size(); i++) {
        if (this->operator[](max) < this->operator[](i)) {
            max = i;
        }
    }

    return max;
}
