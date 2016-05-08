//
// Created by user on 10. 4. 2016.
//

#ifndef NEURONET_ACTORCRITIC_H
#define NEURONET_ACTORCRITIC_H


#include "../../network/NeuralNetwork.h"
#include "TDLambda.h"
#include "../BackProp.h"
#include "CACLAActor.h"
#include "../IEnvironment.h"
#include "QLearning.h"

class ActorCritic {
public:
    ActorCritic(NeuralNetwork* p_actor, NeuralNetwork* p_critic);
    virtual ~ActorCritic();

    void setAlpha(double p_alpha);
    void setBeta(double p_beta);
    void setExploration(double p_epsilon);

    virtual void init(IEnvironment* p_environment);
    virtual void run();

protected:
    virtual void getAction(VectorXd* p_state, VectorXd* p_action);

protected:
    IEnvironment* _environment;
    NeuralNetwork* _actor;
    NeuralNetwork* _critic;

    double _alpha;
    double _beta;
    double _epsilon;

    LearningAlgorithm* _criticLearning;
    LearningAlgorithm* _actorLearning;
};


#endif //NEURONET_ACTORCRITIC_H
