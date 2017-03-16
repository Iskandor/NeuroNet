//
// Created by mpechac on 10. 3. 2017.
//

#ifndef NEURONET_MATRIX_H
#define NEURONET_MATRIX_H


#include "Base.h"
#include "Vector.h"

namespace SFLAB {

class Vector;

class Matrix : public Base {
public:
    Matrix(int p_rows, int p_cols, INIT p_init = ZERO, double p_value = 0);
    Matrix(const Matrix& p_copy);

    void operator = ( const Matrix& p_matrix);
    Matrix operator + ( const Matrix& p_matrix);
    Matrix operator - ( const Matrix& p_matrix);
    Matrix operator * ( const Matrix& p_matrix);
    Vector operator * ( Vector p_vector);
    Matrix operator * ( const double p_const);
    Matrix T();
    Matrix inv();

    inline double *operator [] ( int p_index ) { return _arr[p_index]; };

private:
    void init(INIT p_init, double p_value);

};

}

#endif //NEURONET_MATRIX_H
