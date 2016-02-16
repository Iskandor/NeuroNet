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

    map<int, VectorXd> _gradient;
    vector<NeuralGroup*> _bfsTree;

    void bfsTreeCreate();
    void bfsRecursive(NeuralGroup* p_node);
    void calcGradient();

private:
    void calcDelta(NeuralGroup *p_group);
    void calcGradient(NeuralGroup *p_group);
};


#endif //LIBNEURONET_GRADIENTBASE_H
