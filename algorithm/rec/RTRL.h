//
// Created by mpechac on 12. 7. 2016.
//

#include <GradientDescent.h>

#ifndef NEURONET_RTRL_H
#define NEURONET_RTRL_H


namespace NeuroNet {

class RTRL : public GradientDescent {
    public:
    RTRL(NeuralNetwork* p_network);
    ~RTRL();

    virtual double train(double *p_input, double* p_target);

    private:
    void updateWeights(Connection* p_connection);
    void updateDerivs(Connection* p_connection);

    map<int, Matrix>  _derivs;
};

}


#endif //NEURONET_RTRL_H
