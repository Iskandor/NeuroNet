//
// Created by user on 10. 4. 2016.
//

#ifndef NEURONET_ACTORCRITIC_H
#define NEURONET_ACTORCRITIC_H


#include "../../network/NeuralNetwork.h"
#include "TDLambda.h"
#include "../BackProp.h"
#include "SimpleActor.h"
#include "../IEnvironment.h"

class ActorCritic {
public:
    ActorCritic(NeuralNetwork* p_actor, NeuralNetwork* p_critic);
    virtual ~ActorCritic();

    void setAlpha(double p_alpha);
    void setBeta(double p_beta);

    virtual void init(IEnvironment* p_environment);
    virtual void train();

protected:
    IEnvironment* _environment;
    NeuralNetwork* _actor;
    NeuralNetwork* _critic;

    double _alpha;
    double _beta;

    virtual void getAction(VectorXd* p_state, VectorXd* p_action);

private:
    TDLambda*    _criticLearning;
    SimpleActor* _actorLearning;
};


#endif //NEURONET_ACTORCRITIC_H
