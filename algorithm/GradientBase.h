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

    map<string, VectorXd> _delta;
    vector<NeuralGroup*> _groupTree;

    void groupTreeCreate();
    void bfsRecursive(NeuralGroup* p_node);
    void calcGradient(VectorXd *p_error = nullptr);

private:
    void gradientKernel(NeuralGroup *p_group);
};


#endif //LIBNEURONET_GRADIENTBASE_H
