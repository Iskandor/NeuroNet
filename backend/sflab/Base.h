//
// Created by mpechac on 15. 3. 2017.
//

#ifndef NEURONET_SFLAB_BASE_H
#define NEURONET_SFLAB_BASE_H

#include <iostream>

using namespace std;

namespace SFLAB {

class Base {
public:
    enum INIT {
        ZERO = 0,
        UNITARY = 1,
        ONES = 1,
        VALUE = 2
    };

    Base(int p_rows, int p_cols);

    Base(const Base &p_copy);

    virtual ~Base();

    friend ostream &operator<<(ostream &output, const Base &p_matrix) {
        for (int i = 0; i < p_matrix._rows; i++) {
            for (int j = 0; j < p_matrix._cols; j++) {
                if (j == p_matrix._cols - 1) {
                    output << p_matrix._arr[i][j] << endl;
                }
                else {
                    output << p_matrix._arr[i][j] << ",";
                }
            }
        }

        return output;
    }

    void fill(double p_value);

protected:
    virtual void init(INIT p_init, double p_value) = 0;
    void clone(const Base &p_copy);
    void internal_init();

protected:
    double **_arr = NULL;
    int _rows;
    int _cols;
};
}


#endif //NEURONET_SFLAB_BASE_H
