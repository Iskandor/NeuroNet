#include <iostream>
#include "../log/log.h"
#include "../network/NeuralNetwork.h"
#include "../network/Define.h"
#include "MazeOld.h"
#include "../algorithm/GreedyPolicy.h"
#include "../algorithm/rl/TDLambda.h"

using namespace NeuroNet;

void sampleTD() {
    double sumReward = 0;
    int time = 0;
    int dim = 3;

    NeuralNetwork network;

    NeuralGroup* inputGroup = network.addLayer("input", dim*dim, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    NeuralGroup* biasUnitH = network.addLayer("biasH", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    NeuralGroup* biasUnitO = network.addLayer("biasO", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    NeuralGroup* hiddenGroup = network.addLayer("hidden", 16, NeuralGroup::TANH, NeuralNetwork::HIDDEN);
    NeuralGroup* outputGroup = network.addLayer("output", 1, NeuralGroup::TANH, NeuralNetwork::OUTPUT);


    // feed-forward connections
    network.addConnection(inputGroup, hiddenGroup);
    network.addConnection(hiddenGroup, outputGroup);
    // bias connections
    network.addConnection(biasUnitH, hiddenGroup);
    network.addConnection(biasUnitO, outputGroup);

    TDLambda td(&network, 0.9, 0.9);
    td.setAlpha(.001);

    MazeOld maze(dim);
    maze.reset();

    VectorXd action(4);
    VectorXd state0(dim*dim);
    VectorXd state1(dim*dim);

    GreedyPolicy policy(&network, &maze);
    policy.setEpsilon(0.01);

    FILE* pFile = fopen("application.log", "w");
    Output2FILE::Stream() = pFile;
    FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

    int episode = 0;

    while(episode < 6000) {
        double reward = 0;

        /*
        cout << time << " " << reward << " " << sumReward << " " << endl;
        for(auto i = 0; i < dim; i++) {
            for(auto j = 0; j < dim; j++) {
                cout << (*maze.getState())(i*dim + j);
            }
            cout << endl;
        }
        */

        state0 = *maze.getState();
        policy.getActionV(&state0, &action);
        maze.updateState(&action);
        state1 = *maze.getState();
        reward = maze.getReward();
        sumReward += reward;

        // 3. update
        td.train(&state0, &state1, reward);
        time++;

        // 4. check whether terminal state was reached
        if (maze.isFinished()) {

            /*
            cout << time << " " << reward << " " << sumReward << " " << endl;
            for(auto i = 0; i < dim; i++) {
                for(auto j = 0; j < dim; j++) {
                    cout << (*maze.getState())(i*dim + j);
                }
                cout << endl;
            }
            */

            cout << "Finished episode " << episode << "! " << time << " Reward:" << sumReward << endl;
            FILE_LOG(logDEBUG1) << sumReward;

            maze.reset();
            time = 0;
            sumReward = 0;
            episode++;
        }
    }


    cout << "Uspesne ukoncene." << endl;
}
