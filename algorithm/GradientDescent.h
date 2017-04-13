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
    GradientDescent(NeuralNetwork *p_network);
    virtual ~GradientDescent(void);

    enum GRADIENT {
        REGULAR = 0,
        NATURAL = 1
    };

protected:
    NeuralNetwork*  _network;
    vector<NeuralGroup*> _groupTree;
    map<string, Vector> _delta;
    double _epsilon;

    void groupTreeCreate();

    virtual map<int, Matrix>* calcRegGradient(Vector *p_error);
    map<int, Matrix>* calcNatGradient(double p_epsilon, Vector *p_error);


private:
    map<int, Matrix> _regGradient;
    map<int, Matrix> _natGradient;
    map<int, Matrix> _invFisherMatrix;

    void bfsRecursive(NeuralGroup* p_node);
    void deltaKernel(NeuralGroup *p_group);
    void regGradientKernel(Connection *p_connection);
    void natGradientKernel(Connection *p_connection);
    void invFisherMatrixKernel(Connection *p_connection);

};

}
#endif //LIBNEURONET_GRADIENTBASE_H
