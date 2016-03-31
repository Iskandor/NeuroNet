//
// Created by mpechac on 23. 3. 2016.
//

#include "../network/NeuralNetwork.h"
#include "../network/Define.h"
#include "../algorithm/QLearning.h"
#include "LunarLander.h"
#include "../log/log.h"
#include "../network/som/SOM.h"
#include "../network/filters/NormalizationFilter.h"

void sampleLunarLander() {
    double sumReward = 0;
    int time = 0;
    int dim = 3;
    int episode = 0;
    const int EPISODES = 100000;
    const int SIZE = 9;

    NeuralNetwork network;

    NeuralGroup* inputGroup = network.addLayer("input", dim+2, IDENTITY, NeuralNetwork::INPUT);
    NeuralGroup* biasUnitH = network.addLayer("biasH", 1, BIAS, NeuralNetwork::HIDDEN);
    NeuralGroup* biasUnitO = network.addLayer("biasO", 1, BIAS, NeuralNetwork::HIDDEN);
    NeuralGroup* hiddenGroup = network.addLayer("hidden", 25, SIGMOID, NeuralNetwork::HIDDEN);
    NeuralGroup* contextGroup = network.addLayer("context", 25, SIGMOID, NeuralNetwork::HIDDEN);
    NeuralGroup* outputGroup = network.addLayer("output", 1, IDENTITY, NeuralNetwork::OUTPUT);

    VectorXd limit(dim+2);
    limit << 20,50,20,1,1;
    inputGroup->addInFilter(new NormalizationFilter(&limit));
    // feed-forward connections
    network.addConnection(inputGroup, hiddenGroup);
    network.addConnection(hiddenGroup, outputGroup);
    // bias connections
    network.addConnection(biasUnitH, hiddenGroup);
    network.addConnection(biasUnitO, outputGroup);
    // recurrent connection
    network.addRecConnection(hiddenGroup, contextGroup);
    network.addConnection(contextGroup, hiddenGroup);

    SOM som(dim, SIZE, SIZE, EXPONENTIAL);
    som.initTraining(0.01, EPISODES / 2);

    QLearning qAgent(&network, 0.9, 0.9);
    qAgent.setAlpha(0.1);
    double epsilon = 0.01;

    LunarLander lander;
    VectorXd action(2);
    VectorXd state0(dim);
    VectorXd state1(dim);

    FILE* pFile = fopen("application.log", "w");
    Output2FILE::Stream() = pFile;
    FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

    while(episode < 100000) {
        double maxOutput = -INFINITY;
        int action_i = 0;
        double reward = 0;

        //lander.print(cout);
        //som.train(lander.getState()->data());
        //som.activate(lander.getState());
        //state0 = *som.getOutput();
        state0 = *lander.getState();

        for (int i = 0; i < action.size(); i++) {
            action.fill(0);
            action[i] = 1;

            double roll = rand() % 100;
            if (roll < epsilon * 100) {
                action_i = i;
                break;
            }


            //VectorXd input(SIZE*SIZE + action.size());
            VectorXd input(dim + action.size());
            input << state0, action;
            network.activate(&input);

            if (maxOutput < network.getScalarOutput()) {
                action_i = i;
                maxOutput = network.getScalarOutput();
            }
        }

        action.fill(0);
        action[action_i] = 1;

        lander.updateState(&action);
        //som.activate(lander.getState());
        //state1 = *som.getOutput();
        state1 = *lander.getState();
        reward = lander.getReward();
        sumReward += reward;

        // 3. update
        qAgent.train(&state0, &action, &state1, reward);
        time++;

        // 4. check whether terminal state was reached
        if (lander.isFinished()) {
            cout << "Finished episode " << episode << "! " << time << " Reward:" << sumReward << endl;
            FILE_LOG(logDEBUG1) << sumReward;

            time = 0;
            sumReward = 0;
            lander.reset();
            episode++;
            som.paramDecay();
        }
    }

    cout << "Uspesne ukoncene." << endl;
}
