//
// Created by Matej Pechac on 15. 2. 2016.
//

#ifndef LIBNEURONET_GRADIENTBASE_H
#define LIBNEURONET_GRADIENTBASE_H


#include "../network/NeuralNetwork.h"
#include "LearningAlgorithm.h"

namespace NeuroNet {

class GradientDescent : public LearningAlgorithm {
public:
    GradientDescent(NeuralNetwork *p_network, double p_momentum = 0, bool p_nesterov = false);
    virtual ~GradientDescent(void);

protected:
    vector<NeuralGroup*> _groupTree;
    map<int, MatrixXd> _regGradient;
    map<int, MatrixXd> _natGradient;
    map<int, MatrixXd> _invFisherMatrix;
    map<string, VectorXd> _delta;
    double _epsilon;
    double _momentum;
    bool _nesterov;

    void groupTreeCreate();
    void bfsRecursive(NeuralGroup* p_node);
    void calcRegGradient(VectorXd *p_error);
    void calcNatGradient(double p_epsilon, VectorXd *p_error);
    void deltaKernel(NeuralGroup *p_group);
    void regGradientKernel(Connection *p_connection);
    void natGradientKernel(Connection *p_connection);
    void invFisherMatrixKernel(Connection *p_connection);
};

}
#endif //LIBNEURONET_GRADIENTBASE_H
