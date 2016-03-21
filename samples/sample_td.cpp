#include <iostream>
#include "../log/log.h"
#include "../network/NeuralNetwork.h"
#include "../network/Define.h"
#include "../algorithm/TDLambda.h"
#include "Maze.h"
#include "../algorithm/GreedyPolicy.h"

void sampleTD() {
    double sumReward = 0;
    int time = 0;
    int dim = 3;

    NeuralNetwork network;

    NeuralGroup* inputGroup = network.addLayer("input", dim*dim, IDENTITY, NeuralNetwork::INPUT);
    NeuralGroup* biasUnitH = network.addLayer("biasH", 1, BIAS, NeuralNetwork::HIDDEN);
    NeuralGroup* biasUnitO = network.addLayer("biasO", 1, BIAS, NeuralNetwork::HIDDEN);
    NeuralGroup* hiddenGroup = network.addLayer("hidden", 9, TANH, NeuralNetwork::HIDDEN);
    NeuralGroup* outputGroup = network.addLayer("output", 1, IDENTITY, NeuralNetwork::OUTPUT);


    // feed-forward connections
    network.addConnection(inputGroup, hiddenGroup);
    network.addConnection(hiddenGroup, outputGroup);
    // bias connections
    network.addConnection(biasUnitH, hiddenGroup);
    network.addConnection(biasUnitO, outputGroup);

    TDLambda td(&network, 0.9, 0.9);
    td.setAlpha(.1);

    Maze maze(dim);
    maze.reset();

    VectorXd action(4);
    VectorXd state0(dim*dim);
    VectorXd state1(dim*dim);

    GreedyPolicy policy(&network, &maze);
    policy.setEpsilon(0.001);

    FILE* pFile = fopen("application.log", "w");
    Output2FILE::Stream() = pFile;
    FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

    int episode = 0;

    while(episode < 200) {
        double reward = 0;
        state0 = *maze.getState();
        policy.getAction(action, dim*dim);
        maze.updateState(&action);
        state1 = *maze.getState();
        reward = maze.getReward();
        sumReward += reward;

        // 3. update
        td.train(&state0, &state1, reward);
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

            cout << "Finished episode" << episode << "! " << time << " Reward:" << sumReward << endl;
            FILE_LOG(logDEBUG1) << sumReward;
            for(auto i = 0; i < dim; i++) {
              for(auto j = 0; j < dim; j++) {
                state0.fill(0);
                state0[i*dim + j] = 1;
                network.setInput(&state0);
                network.onLoop();
                cout << network.getScalarOutput() << ",";
              }
              cout << endl;
            }

            time = 0;
            sumReward = 0;
            maze.reset();
            episode++;
        }
    }


    cout << "Uspesne ukoncene." << endl;
}
