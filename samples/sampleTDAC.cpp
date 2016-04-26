#include <iostream>
#include "../log/log.h"
#include "../network/NeuralNetwork.h"
#include "../network/Define.h"
#include "Maze.h"
#include "../algorithm/rl/QLearning.h"
#include "../algorithm/rl/ActorCritic.h"

void sampleTDAC() {
    double sumReward = 0;
    int time = 0;
    int dim = 3;

    NeuralNetwork critic;
    critic.addLayer("input", dim*dim, IDENTITY, NeuralNetwork::INPUT);
    critic.addLayer("biasH", 1, BIAS, NeuralNetwork::HIDDEN);
    critic.addLayer("biasO", 1, BIAS, NeuralNetwork::HIDDEN);
    critic.addLayer("hidden", 9, TANH, NeuralNetwork::HIDDEN);
    critic.addLayer("output", 1, IDENTITY, NeuralNetwork::OUTPUT);
    // feed-forward connections
    critic.addConnection("input", "hidden");
    critic.addConnection("hidden", "output");
    // bias connections
    critic.addConnection("biasH", "hidden");
    critic.addConnection("biasO", "output");

    NeuralNetwork actor;
    actor.addLayer("input", dim*dim, IDENTITY, NeuralNetwork::INPUT);
    actor.addLayer("biasH", 1, BIAS, NeuralNetwork::HIDDEN);
    actor.addLayer("biasO", 1, BIAS, NeuralNetwork::HIDDEN);
    actor.addLayer("hidden", 9, TANH, NeuralNetwork::HIDDEN);
    actor.addLayer("output", 4, SOFTMAX, NeuralNetwork::OUTPUT);
    actor.addConnection("input", "hidden");
    actor.addConnection("hidden", "output");
    actor.addConnection("biasH", "hidden");
    actor.addConnection("biasO", "output");
    //actor.getGroup("output")->addOutFilter(new KwtaFilter(1, true));


    Maze maze(dim);
    maze.reset();

    ActorCritic actorCritic(&actor, &critic);
    actorCritic.setAlpha(0.01);
    actorCritic.setBeta(0.1);
    actorCritic.init(&maze);


    FILE* pFile = fopen("application.log", "w");
    Output2FILE::Stream() = pFile;
    FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

    int episode = 0;

    VectorXd state0(dim*dim);

    while(episode < 200) {

        actorCritic.train();

        sumReward += maze.getReward();
        time++;

        // 4. check whether terminal state was reached
        if (maze.isFinished()) {

            cout << "Finished episode " << episode << "! " << time << " Reward:" << sumReward << endl;
            FILE_LOG(logDEBUG1) << sumReward;
            for(auto i = 0; i < dim; i++) {
              for(auto j = 0; j < dim; j++) {
                state0.fill(0);
                state0[i*dim + j] = 1;
                critic.activate(&state0);
                cout << critic.getScalarOutput() << ",";
              }
              cout << endl;
            }

            time = 0;
            sumReward = 0;
            episode++;
        }
    }


    cout << "Uspesne ukoncene." << endl;
}
