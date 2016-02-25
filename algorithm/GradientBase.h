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

    map<int, VectorXd> _delta;
    map<int, MatrixXd> _gradient;
    vector<NeuralGroup*> _groupTree;

    void groupTreeCreate();
    void bfsRecursive(NeuralGroup* p_node);
    virtual void calcGradient(VectorXd* p_error);

private:
    void calcDelta(NeuralGroup *p_group);
    void calcGradient(Connection *p_connection);
};


#endif //LIBNEURONET_GRADIENTBASE_H
