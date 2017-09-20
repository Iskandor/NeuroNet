//
// Created by mpechac on 15. 3. 2017.
//

#ifndef NEURONET_FLAB_BASE_H
#define NEURONET_FLAB_BASE_H

#include <iostream>

using namespace std;

namespace FLAB {

class Base {
public:
    enum INIT {
        ZERO = 0,
        IDENTITY = 1,
        ONES = 1,
        VALUE = 2,
        RANDOM = 3
    };

    Base(int p_rows = 0, int p_cols = 0);
    Base(int p_rows, int p_cols, double* p_data);
    Base(int p_rows, int p_cols, initializer_list <double> p_inputs);
    Base(const Base &p_copy);

    virtual ~Base();

    friend ostream &operator<<(ostream &output, const Base &p_base) {
        for (int i = 0; i < p_base._rows; i++) {
            for (int j = 0; j < p_base._cols; j++) {
                if (j == p_base._cols - 1) {
                    output << p_base._arr[i * p_base._cols + j] << endl;
                }
                else {
                    output << p_base._arr[i * p_base._cols + j] << ",";
                }
            }
        }

        return output;
    }

    void fill(double p_value);

    double maxCoeff();
    double minCoeff();

    inline int rows() { return _rows; };
    inline int cols() { return _cols; };

protected:
    static double* allocBuffer(int p_rows, int p_cols);
    virtual void init(INIT p_init, double p_value) = 0;
    void clone(const Base &p_copy);
    void internal_init(double *p_data = NULL);
    void internal_init(initializer_list <double> p_inputs);

protected:
    double *_arr = NULL;
    int _rows;
    int _cols;
};
}


#endif //NEURONET_FLAB_BASE_H
