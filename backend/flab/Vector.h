//
// Created by mpechac on 15. 3. 2017.
//

#ifndef NEURONET_VECTOR_H
#define NEURONET_VECTOR_H

#include <iterator>
#include "Base.h"
#include "Matrix.h"

namespace FLAB {

class Matrix;

class Vector : public Base {
public:
    static Vector Zero(int p_dim);
    static Vector One(int p_dim);
    static Vector Random(int p_dim);
    static Vector Concat(Vector& p_vector1, Vector& p_vector2);

    Vector(int p_dim = 0, const INIT &p_init = ZERO, double p_value = 0);
    Vector(int p_dim, double* p_data);
    Vector(int p_rows, int p_cols, double* p_data);
    Vector(int p_dim, std::initializer_list <double> inputs);
    Vector(int p_rows, int p_cols, const INIT &p_init = ZERO, double p_value = 0);
    Vector(const Vector& p_copy);
    ~Vector();

    void operator = ( const Vector& p_vector);
    Vector operator + ( const Vector& p_vector);
    void operator += ( const Vector& p_vector);
    Vector operator - ( const Vector& p_vector);
    void operator -= ( const Vector& p_vector);
    Matrix operator * ( const Vector& p_vector);
    Vector operator * ( const double p_const);
    void operator *= ( const double p_const);

    friend Vector operator * ( const double p_const, const Vector& p_vector) {
        if (p_vector._cols == 1) {
            Vector res(p_vector._rows);

            for (int i = 0; i < p_vector._rows; i++) {
                res._arr[i] = p_const * p_vector._arr[i];
            }

            return Vector(res);
        }
        else if (p_vector._rows == 1) {
            Vector res(p_vector._cols);

            for (int i = 0; i < p_vector._cols; i++) {
                res._arr[i] = p_const * p_vector._arr[i];
            }

            return Vector(res);
        }

        return Vector();
    }

    Vector T();
    double norm();

    int minIndex();
    int maxIndex();

    double& operator [] ( int p_index );

    inline int size() { return _rows * _cols; };

private:
    void init(INIT p_init, double p_value);
};

}

#endif //NEURONET_VECTOR_H
