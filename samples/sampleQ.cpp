//
// Created by user on 25. 2. 2016.
//

#include <iostream>
#include "../network/NeuralNetwork.h"
#include "../network/Define.h"
#include "../environments/maze/Maze.h"
#include "../log/log.h"
#include "../algorithm/rl/QLearning.h"
#include "../network/NetworkUtils.h"
#include "../network/RandomGenerator.h"

#include "MazeTask.h"

using namespace NeuroNet;

VectorXd encodeState(vector<int> *p_sensors) {
    VectorXd res(64);
    VectorXd encoded(4);

    for(unsigned int i = 0; i < p_sensors->size(); i++) {
        if (p_sensors->at(i) > 0) {
            NetworkUtils::binaryEncoding(p_sensors->at(i) - 1, &encoded);
        }
        else {
            encoded.fill(1);
        }

        for(int j = 0; j < 4; j++) {
            res[i * 4 + j] = encoded[j];
        }
    }

    return VectorXd(res);
}

int chooseAction(VectorXd* p_input) {
    int maxI = 0;

    for(int i = 0; i < p_input->size(); i++) {
        if ((*p_input)[i] > (*p_input)[maxI]) {
            maxI = i;
        }
    }

    return maxI;
}

void sampleQ2() {
    MazeTask task;
    Maze* maze = task.getEnvironment();

    NeuralNetwork network;

    network.addLayer("input", 64, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    network.addLayer("hidden0", 164, NeuralGroup::RELU, NeuralNetwork::HIDDEN);
    network.addLayer("hidden1", 150, NeuralGroup::RELU, NeuralNetwork::HIDDEN);
    network.addLayer("output", 4, NeuralGroup::LINEAR, NeuralNetwork::OUTPUT);

    // feed-forward connections
    network.addConnection("input", "hidden0");
    network.addConnection("hidden0", "hidden1");
    network.addConnection("hidden1", "output");

    RMSProp optimizer(&network, 0.9, 1e-6, 0.9);
    //BackProp optimizer(&network, 1e-6, 0.9, true);
    QLearning agent(&optimizer, &network, 0.9, 0);

    agent.setAlpha(0.001);
    agent.setBatchSize(10);

    vector<int> sensors;
    VectorXd state0, state1;
    int action;
    double reward = 0;
    double epsilon = 1;
    double random;
    int epochs = 1000;

    int wins = 0, loses = 0;

    //FILE* pFile = fopen("application.log", "w");
    //Output2FILE::Stream() = pFile;
    //FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

    for (int e = 0; e < epochs; e++) {
        cout << "Epoch " << e << endl;

        task.getEnvironment()->reset();

        while(!task.isFinished()) {
            //cout << maze->toString() << endl;

            sensors = maze->getSensors();
            state0 = encodeState(&sensors);
            network.activate(&state0);

            random = RandomGenerator::getInstance().random();

            if (random < epsilon) {
                action = RandomGenerator::getInstance().random(0, 3);
            }
            else {
                action = chooseAction(network.getOutput());
            }
            maze->performAction(action);

            sensors = maze->getSensors();
            state1 = encodeState(&sensors);
            reward = task.getReward();
            agent.train(&state0, action, &state1, reward);
        }

        if (reward > 0) {
            wins++;
        }
        else {
            loses++;
        }

        cout << maze->toString() << endl;
        cout << wins << " / " << loses << endl;
        //FILE_LOG(logDEBUG1) << (double)wins / (double)loses;


        if (epsilon > 0.1) {
            epsilon -= (1.0 / epochs);
        }
    }
}