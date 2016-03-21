//
// Created by Matej Pechac on 15. 2. 2016.
//

#ifndef LIBNEURONET_GRADIENTBASE_H
#define LIBNEURONET_GRADIENTBASE_H


#include "../network/NeuralNetwork.h"

class GradientBase {
public:
    GradientBase(NeuralNetwork *p_network);
    virtual ~GradientBase(void);

protected:
    NeuralNetwork* _network;
    vector<NeuralGroup*> _groupTree;
    map<int, MatrixXd> _gradient;
    map<string, VectorXd> _delta;

    void groupTreeCreate();
    void bfsRecursive(NeuralGroup* p_node);
    void calcGradient(VectorXd *p_error = nullptr);

    virtual void deltaKernel(NeuralGroup *p_group);
    void gradientKernel(Connection *p_connection);
};


#endif //LIBNEURONET_GRADIENTBASE_H
