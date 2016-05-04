//
// Created by user on 1. 5. 2016.
//

#ifndef NEURONET_LEARNINGALGORITHM_H
#define NEURONET_LEARNINGALGORITHM_H


class LearningAlgorithm {

public:
    LearningAlgorithm();
    virtual ~LearningAlgorithm();

    void setAlpha(double p_alpha);
protected:
    double  _alpha;
};


#endif //NEURONET_LEARNINGALGORITHM_H
