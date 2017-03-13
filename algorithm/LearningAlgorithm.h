//
// Created by user on 1. 5. 2016.
//

#ifndef NEURONET_LEARNINGALGORITHM_H
#define NEURONET_LEARNINGALGORITHM_H


#include "../network/NeuralNetwork.h"

namespace NeuroNet {

class LearningAlgorithm {

public:
    LearningAlgorithm();
    virtual ~LearningAlgorithm();

    virtual void setAlpha(double p_alpha);
    virtual void setBatchSize(int p_batchSize);

protected:
    void    updateBatch();

protected:
    double  _alpha;
    int     _batchSize;
    int     _batch;

};

}
#endif //NEURONET_LEARNINGALGORITHM_H
