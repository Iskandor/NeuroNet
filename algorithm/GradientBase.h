//
// Created by Matej Pechac on 15. 2. 2016.
//

#ifndef LIBNEURONET_GRADIENTBASE_H
#define LIBNEURONET_GRADIENTBASE_H


#include "../network/NeuralNetwork.h"

namespace NeuroNet {

class GradientBase {
public:
    GradientBase(NeuralNetwork *p_network);
    virtual ~GradientBase(void);

protected:
    NeuralNetwork* _network;
    vector<NeuralGroup*> _groupTree;
    map<int, MatrixXd> _regGradient;
    map<int, MatrixXd> _natGradient;
    map<int, MatrixXd> _invFisherMatrix;
    double _epsilon;

    void groupTreeCreate();
    void bfsRecursive(NeuralGroup* p_node);
    void calcRegGradient(VectorXd *p_error = nullptr);
    void calcNatGradient(double p_epsilon, VectorXd *p_error = nullptr);

    void regGradientKernel(Connection *p_connection);
    void natGradientKernel(Connection *p_connection);
    void invFisherMatrixKernel(Connection *p_connection);
private:
    map<string, VectorXd> _delta;
    void deltaKernel(NeuralGroup *p_group);
};

}
#endif //LIBNEURONET_GRADIENTBASE_H
