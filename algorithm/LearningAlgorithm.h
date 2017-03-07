//
// Created by user on 1. 5. 2016.
//

#ifndef NEURONET_LEARNINGALGORITHM_H
#define NEURONET_LEARNINGALGORITHM_H


#include "../network/NeuralNetwork.h"

namespace NeuroNet {

class LearningAlgorithm {

public:
    LearningAlgorithm(NeuralNetwork* p_network);
    virtual ~LearningAlgorithm();

    void setAlpha(double p_alpha);
    void setBatchSize(int p_batchSize);

protected:
    NeuralNetwork* _network;
    map<int, MatrixXd> _weightDelta;
    map<int, VectorXd> _biasDelta;

    double  _alpha;
    int     _batchSize;
    int     _batch;

};

}
#endif //NEURONET_LEARNINGALGORITHM_H
