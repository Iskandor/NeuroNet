//
// Created by mpechac on 15. 3. 2017.
//

#ifndef NEURONET_VECTOR_H
#define NEURONET_VECTOR_H

#include "Base.h"
#include "Matrix.h"

namespace SFLAB {

class Matrix;

class Vector : public Base {
public:
    Vector(int p_dim, const INIT &p_init = ZERO, double p_value = 0);
    Vector(int p_rows, int p_cols, const INIT &p_init = ZERO, double p_value = 0);
    Vector(const Vector& p_copy);
    ~Vector();

    void operator = ( const Vector& p_vector);
    Vector operator + ( const Vector& p_vector);
    Vector operator - ( const Vector& p_vector);
    Matrix operator * ( const Vector& p_vector);
    Vector operator * ( const double p_const);
    Vector T();

    double& operator [] ( int p_index );

private:
    void init(INIT p_init, double p_value);
};

}

#endif //NEURONET_VECTOR_H
