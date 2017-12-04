//
// Created by mpechac on 10. 3. 2017.
//

#ifndef NEURONET_MATRIX_H
#define NEURONET_MATRIX_H


#include "Base.h"
#include "Vector.h"
#include "Tensor3.h"

namespace FLAB {

class Vector;

class Matrix : public Base {
public:
    static Matrix Zero(int p_rows, int p_cols);
    static Matrix Random(int p_rows, int p_cols);
    static Matrix Identity(int p_rows, int p_cols);
    static Matrix Value(int p_rows, int p_cols, double p_value);

    Matrix(int p_rows = 0, int p_cols = 0, INIT p_init = ZERO, double p_value = 0);
    Matrix(int p_rows, int p_cols, double* p_data);
    Matrix(int p_rows, int p_cols, std::initializer_list <double> p_inputs);
    Matrix(const Matrix& p_copy);

    void operator = ( const Matrix& p_matrix);
    Matrix operator + ( const Matrix& p_matrix);
    void operator += ( const Matrix& p_matrix);
    Matrix operator - ( const Matrix& p_matrix);
    void operator -= ( const Matrix& p_matrix);
    Matrix operator * ( const Matrix& p_matrix);
    Vector operator * ( Vector p_vector);
    Matrix operator * ( const double p_const);
    void operator *= ( const double p_const);

    friend Matrix operator * ( const double p_const, const Matrix& p_matrix) {
        Matrix res(p_matrix._rows, p_matrix._cols);

        for(int i = 0; i < p_matrix._rows; i++) {
            for(int j = 0; j < p_matrix._cols; j++) {
                res._arr[i * p_matrix._cols + j] = p_const * p_matrix._arr[i * p_matrix._cols + j];
            }
        }

        return Matrix(res);
    }

    Matrix T();
    Matrix inv();
    Matrix ew_sqrt();
    Matrix ew_pow(int p_n);
    Matrix ew_dot(const Matrix &p_matrix);

    inline double at(int p_row, int p_col) { return _arr[p_row * _cols + p_col]; };
    inline double set(int p_row, int p_col, double p_value) { _arr[p_row * _cols + p_col] = p_value; };

private:
    void init(INIT p_init, double p_value);

};

}

#endif //NEURONET_MATRIX_H
