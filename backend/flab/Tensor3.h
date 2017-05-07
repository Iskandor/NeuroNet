//
// Created by mpechac on 28. 3. 2017.
//

#ifndef NEURONET_TENSOR_H
#define NEURONET_TENSOR_H


#include <vector>
#include "Base.h"

using namespace std;

namespace FLAB {

class Tensor3 {
public:
    enum INIT {
        ZERO = 0,
        ONES = 1,
        VALUE = 2,
        RANDOM = 3
    };

    static Tensor3 Zero(int p_x, int p_y, int p_z);
    static Tensor3 Random(int p_x, int p_y, int p_z);
    static Tensor3 Ones(int p_x, int p_y, int p_z);
    static Tensor3 Value(int p_x, int p_y, int p_z, double p_value);

    Tensor3(int p_x = 0, int p_y = 0, int p_z = 0, INIT p_init = ZERO, double p_value = 0);
    Tensor3(int p_x, int p_y, int p_z, double* p_data);
    Tensor3(const Tensor3 &p_copy);

    virtual ~Tensor3();

    double operator()(int x, int y, int z);
    void set(int x, int y, int z, double p_value);

    friend ostream &operator<<(ostream &output, const Tensor3 &p_base) {
        return output;
    }

    void fill(double p_value);
    inline int dim(int p_index) { return _dims[p_index]; };

protected:
    void init(INIT p_init, double p_value);
    void clone(const Tensor3 &p_copy);

protected:
    double *_arr = NULL;
    vector<int> _dims;
    int _dim;

private:
    int calcDim();
};

}

#endif //NEURONET_TENSOR_H
