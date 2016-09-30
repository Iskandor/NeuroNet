//
// Created by mpechac on 12. 7. 2016.
//

#include "../GradientBase.h"
#include "../LearningAlgorithm.h"

#ifndef NEURONET_RTRL_H
#define NEURONET_RTRL_H


namespace NeuroNet {

class RTRL : public GradientBase, public LearningAlgorithm {
    public:
    RTRL(NeuralNetwork* p_network);
    ~RTRL();

    virtual double train(double *p_input, double* p_target);

    private:
    void updateWeights(Connection* p_connection);
    void updateDerivs(Connection* p_connection);

    map<int, MatrixXd>  _derivs;
};

}


#endif //NEURONET_RTRL_H
