//
// Created by mpechac on 10. 6. 2016.
//

#include "PoleCart.h"
#include "../network/NeuralNetwork.h"
#include "../algorithm/rl/CACLA.h"
#include "../log/log.h"
#include "../network/NetworkUtils.h"

using namespace NeuroNet;

void samplePoleCart() {
    PoleCart poleCart;
    int time = 0;
    int dim = poleCart.getStateSize();
    int episode = 0;
    double sumReward = 0;

    NeuralNetwork *critic = new NeuralNetwork();
    critic->addLayer("input", dim, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    critic->addLayer("biasH", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    critic->addLayer("biasO", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    critic->addLayer("hidden", 80, NeuralGroup::TANH, NeuralNetwork::HIDDEN);
    //critic->addLayer("context", 50, NeuralGroup::IDENTITY, NeuralNetwork::HIDDEN);
    critic->addLayer("output", 1, NeuralGroup::TANH, NeuralNetwork::OUTPUT);

    // feed-forward connections
    critic->addConnection("input", "hidden");
    critic->addConnection("hidden", "output");
    // bias connections
    critic->addConnection("biasH", "hidden");
    critic->addConnection("biasO", "output");
    // recurrent connections
    //critic->addRecConnection("hidden", "context");
    //critic->addConnection("context", "hidden");

    NeuralNetwork *actor = new NeuralNetwork();
    actor->addLayer("input", dim, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    actor->addLayer("biasH", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    actor->addLayer("biasO", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    actor->addLayer("hidden", 80, NeuralGroup::TANH, NeuralNetwork::HIDDEN);
    //actor->addLayer("context", 20, NeuralGroup::IDENTITY, NeuralNetwork::HIDDEN);
    actor->addLayer("output", 1, NeuralGroup::TANH, NeuralNetwork::OUTPUT);
    // feed-forward connections
    actor->addConnection("input", "hidden");
    actor->addConnection("hidden", "output");
    // bias connections
    actor->addConnection("biasH", "hidden");
    actor->addConnection("biasO", "output");
    // recurrent connections
    //actor->addRecConnection("hidden", "context");
    //actor->addConnection("context", "hidden");


    //NeuralNetwork *actor = NetworkUtils::loadNetwork("polecart_actor.net");
    //NeuralNetwork *critic = NetworkUtils::loadNetwork("polecart_ciritc.net");

    CACLA agent(actor, critic);

    //CACLA s TD hidden 200 TANH
    agent.setCriticStepSize(0.0001);
    agent.setActorStepSize(0.00005); //
    agent.setExploration(0.01); //

    /* CACLA so SARSA
    agent.setCriticStepSize(0.0001);
    agent.setActorStepSize(0.00009);
    agent.setExploration(0.01);
     */
    agent.init(&poleCart);

    FILE* pFile = fopen("application.log", "w");
    Output2FILE::Stream() = pFile;
    FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

    /*
    VectorXd action = VectorXd::Zero(1);
    for(int i = 0; i < 1000; i++) {
        if (i == 0) {
            action[0] = -0.1;
        }
        else {
            action[0] = 0;
        }
        poleCart.updateState(&action);
        cout << i << endl;
        poleCart.print(cout);
        FILE_LOG(logDEBUG1) << poleCart.getTheta();
    }
    */

    while(episode < 10000) {

        agent.run();
        sumReward += poleCart.getReward();

        // 3. update
        time++;

        /*
        cout << "hidden" << endl;
        cout << *critic->getGroup("hidden")->getOutput() << endl;
        cout << "context" << endl;
        cout << *critic->getGroup("context")->getOutput() << endl;
         */

        // 4. check whether terminal state was reached
        if (poleCart.isFinished()) {
            cout << "Finished episode " << episode << "! " << time << " Reward:" << sumReward << endl;
            FILE_LOG(logDEBUG1) << sumReward;

            time = 0;
            sumReward = 0;
            poleCart.reset();
            episode++;
        }
        if (poleCart.isFailed()) {
            cout << "Failed episode " << episode << "! " << time << " Reward:" << sumReward << endl;
            FILE_LOG(logDEBUG1) << sumReward;

            time = 0;
            sumReward = 0;
            poleCart.reset();
            episode++;
        }
    }

    NetworkUtils::saveNetwork("polecart_actor.net", actor);
    NetworkUtils::saveNetwork("polecart_ciritc.net", critic);

    delete actor;
    delete critic;

    cout << "Uspesne ukoncene." << endl;


}