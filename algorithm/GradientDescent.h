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

    enum GRADIENT {
        REGULAR = 0,
        NATURAL = 1
    };

protected:
    NeuralNetwork*  _network;
    vector<NeuralGroup*> _groupTree;
    map<string, VectorXd> _delta;
    double _epsilon;
    double _momentum;
    bool   _nesterov;

    void groupTreeCreate();

    map<int, MatrixXd>* calcRegGradient(VectorXd *p_error);
    map<int, MatrixXd>* calcNatGradient(double p_epsilon, VectorXd *p_error);


private:
    map<int, MatrixXd> _regGradient;
    map<int, MatrixXd> _natGradient;
    map<int, MatrixXd> _invFisherMatrix;

    void bfsRecursive(NeuralGroup* p_node);
    void deltaKernel(NeuralGroup *p_group);
    void regGradientKernel(Connection *p_connection);
    void natGradientKernel(Connection *p_connection);
    void invFisherMatrixKernel(Connection *p_connection);

};

}
#endif //LIBNEURONET_GRADIENTBASE_H
