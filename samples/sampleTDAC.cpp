#include <iostream>
#include "../log/log.h"
#include "../network/NeuralNetwork.h"
#include "../network/Define.h"
#include "Maze.h"
#include "../algorithm/GreedyPolicy.h"
#include "../algorithm/rl/TDLambda.h"
#include "../network/filters/KwtaFilter.h"
#include "../algorithm/rl/Actor.h"
#include "../algorithm/rl/QLearning.h"

void sampleTDAC() {
    double sumReward = 0;
    int time = 0;
    int dim = 3;

    NeuralNetwork critic;
    critic.addLayer("input", 4+dim*dim, IDENTITY, NeuralNetwork::INPUT);
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

    QLearning criticQ(&critic, 0.9, 0.9);
    criticQ.setAlpha(.1);
    Actor actorTD(&actor);
    actorTD.setAlpha(.1);

    Maze maze(dim);
    maze.reset();

    double tdError = 0;
    VectorXd action(4);
    VectorXd state0(dim*dim);
    VectorXd state1(dim*dim);

    //GreedyPolicy policy(&critic, &maze);
    //policy.setEpsilon(0.01);

    FILE* pFile = fopen("application.log", "w");
    Output2FILE::Stream() = pFile;
    FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

    int episode = 0;

    while(episode < 200) {
        double reward = 0;
        state0 = *maze.getState();
        //policy.getActionV(&state0, &action);
        actorTD.getAction(&state0, &action);
        maze.updateState(&action);
        state1 = *maze.getState();
        reward = maze.getReward();
        sumReward += reward;

        // 3. update
        tdError = criticQ.train(&state0, &action, &state1, reward);
        actorTD.train(&state0, tdError);
        time++;

        /*
        cout << time << " " << reward << " " << action_i << " " <<  network.getScalarOutput() << endl;
        for(auto i = 0; i < dim; i++) {
        for(auto j = 0; j < dim; j++) {
          cout << (*maze.getState())(i*dim + j);
        }
        cout << endl;
        }
        */

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
