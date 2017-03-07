#include <iostream>
#include "../log/log.h"
#include "../network/NeuralNetwork.h"
#include "../network/Define.h"
#include "MazeOld.h"
#include "../algorithm/rl/QLearning.h"
#include "../algorithm/rl/ActorCritic.h"
#include "../algorithm/rl/CACLA.h"
#include "../algorithm/rl/RGAC.h"
#include "../network/NetworkUtils.h"
#include "../algorithm/rl/NGAC.h"

using namespace NeuroNet;

void sampleTDAC() {
    double sumReward = 0;
    int time = 0;
    int dim = 3;


    NeuralNetwork* critic = new NeuralNetwork();
    critic->addLayer("input", 4+dim*dim, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    critic->addLayer("biasH", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    critic->addLayer("biasO", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    critic->addLayer("hidden", 32, NeuralGroup::TANH, NeuralNetwork::HIDDEN);
    critic->addLayer("output", 1, NeuralGroup::TANH, NeuralNetwork::OUTPUT);
    // feed-forward connections
    critic->addConnection("input", "hidden");
    critic->addConnection("hidden", "output");
    // bias connections
    critic->addConnection("biasH", "hidden");
    critic->addConnection("biasO", "output");


    NeuralNetwork* actor = new NeuralNetwork();
    actor->addLayer("input", dim*dim, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    actor->addLayer("biasH", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    actor->addLayer("biasO", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    actor->addLayer("hidden", 50, NeuralGroup::SIGMOID, NeuralNetwork::HIDDEN);
    actor->addLayer("output", 4, NeuralGroup::SIGMOID, NeuralNetwork::OUTPUT);
    actor->addConnection("input", "hidden");
    actor->addConnection("hidden", "output");
    actor->addConnection("biasH", "hidden");
    actor->addConnection("biasO", "output");
    //actor.getGroup("output")->addOutFilter(new KwtaFilter(1, true));

    /*
    NetworkUtils::saveNetwork("cacla_actor.net", actor);
    NetworkUtils::saveNetwork("calca_ciritc.net", critic);


    NeuralNetwork* actor = NetworkUtils::loadNetwork("cacla_actor.net");
    NeuralNetwork* critic = NetworkUtils::loadNetwork("calca_ciritc.net");
    */

    MazeOld maze(dim);
    maze.reset();

    RGAC actorCritic(actor, critic);
    actorCritic.setCriticStepSize(0.0001);
    actorCritic.setActorStepSize(.1);
    actorCritic.setExploration(0.01);
    actorCritic.init(&maze);


    FILE* pFile = fopen("application.log", "w");
    Output2FILE::Stream() = pFile;
    FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

    int episode = 0;

    while(episode < 2000) {

        actorCritic.run();

        sumReward += maze.getReward();
        time++;

        // 4. check whether terminal state was reached
        if (maze.isFinished()) {

            cout << "Finished episode " << episode << "! " << time << " Reward:" << sumReward << endl;
            FILE_LOG(logDEBUG1) << sumReward;

            maze.reset();
            time = 0;
            sumReward = 0;
            episode++;
        }
    }

    NetworkUtils::saveNetwork("tdac_actor.net", actor);
    NetworkUtils::saveNetwork("tdac_ciritc.net", critic);

    delete actor;
    delete critic;

    cout << "Uspesne ukoncene." << endl;
}
