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
    int time = 0;
    int dim = 148;
    int episode = 0;
    double sumReward = 0;

    NeuralNetwork *critic = new NeuralNetwork();
    critic->addLayer("input", dim+1, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    critic->addLayer("biasH", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    critic->addLayer("biasO", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    critic->addLayer("hidden", 40, NeuralGroup::TANH, NeuralNetwork::HIDDEN);
    critic->addLayer("output", 1, NeuralGroup::TANH, NeuralNetwork::OUTPUT);

    // feed-forward connections
    critic->addConnection("input", "hidden", 0.3);
    critic->addConnection("hidden", "output", 0.3);
    // bias connections
    critic->addConnection("biasH", "hidden", 0.3);
    critic->addConnection("biasO", "output", 0.3);

    NeuralNetwork *actor = new NeuralNetwork();
    actor->addLayer("input", dim, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    actor->addLayer("biasH", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    actor->addLayer("biasO", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    actor->addLayer("hidden", 40, NeuralGroup::TANH, NeuralNetwork::HIDDEN);
    actor->addLayer("output", 1, NeuralGroup::TANH, NeuralNetwork::OUTPUT);
    // feed-forward connections
    actor->addConnection("input", "hidden", 0.3);
    actor->addConnection("hidden", "output", 0.3);
    // bias connections
    actor->addConnection("biasH", "hidden", 0.3);
    actor->addConnection("biasO", "output", 0.3);

    CACLA agent(actor, critic);
    PoleCart poleCart;

    agent.setAlpha(0.001);
    agent.setBeta(0.001);
    agent.setExploration(0.3);
    agent.init(&poleCart);

    FILE* pFile = fopen("application.log", "w");
    Output2FILE::Stream() = pFile;
    FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

    while(episode < 10000) {

        agent.run();
        sumReward += poleCart.getReward();

        // 3. update
        time++;

        // 4. check whether terminal state was reached
        if (poleCart.isFinished()) {
            cout << "Finished episode " << episode << "! " << time << " Reward:" << sumReward << endl;
            //FILE_LOG(logDEBUG1) << sumReward;

            time = 0;
            sumReward = 0;
            poleCart.reset();
            episode++;
        }
        if (poleCart.isFailed()) {
            cout << "Failed episode " << episode << "! " << time << " Reward:" << sumReward << endl;
            //FILE_LOG(logDEBUG1) << sumReward;

            time = 0;
            sumReward = 0;
            poleCart.reset();
            episode++;
        }
    }

    NetworkUtils::saveNetwork("polecart_actor.net", actor);
    NetworkUtils::saveNetwork("polecart_ciritc.net", critic);


    cout << "Uspesne ukoncene." << endl;


}