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

#include "MazeTask.h"
#include "../algorithm/optimizer/RMSProp.h"
#include "../algorithm/rl/SARSA.h"
#include "../algorithm/optimizer/ADAM.h"
#include "../algorithm/rl/ActorLearning.h"
#include "../algorithm/optimizer/BackProp.h"
#include "../backend/sflab/RandomGenerator.h"

using namespace NeuroNet;

Vector encodeState(vector<int> *p_sensors) {
    Vector res(64);
    Vector encoded(4);

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

    return Vector(res);
}

int chooseAction(Vector* p_input, double epsilon) {

    int action = 0;
    double random = RandomGenerator::getInstance().random();

    if (random < epsilon) {
        action = RandomGenerator::getInstance().random(0, 3);
    }
    else {
        for(int i = 0; i < p_input->size(); i++) {
            if ((*p_input)[i] > (*p_input)[action]) {
                action = i;
            }
        }
    }

    return action;
}

void sampleQ() {
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

    ADAM optimizer(&network);
    //BackProp optimizer(&network, 1e-6, 0.9, true, GradientDescent::NATURAL);
    QLearning agent(&optimizer, &network, 0.9, 0);

    agent.setAlpha(0.001);
    //agent.setBatchSize(10);

    vector<int> sensors;
    Vector state0, state1;
    int action;
    double reward = 0;
    double epsilon = 1;
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
            action = chooseAction(network.getOutput(), epsilon);
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

        //cout << maze->toString() << endl;
        cout << wins << " / " << loses << endl;
        //FILE_LOG(logDEBUG1) << (double)wins / (double)loses;


        if (epsilon > 0.1) {
            epsilon -= (1.0 / epochs);
        }
    }
}

void sampleSARSA() {
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

    ADAM optimizer(&network);
    //BackProp optimizer(&network, 1e-6, 0.9, true, GradientDescent::NATURAL);
    SARSA agent(&optimizer, &network, 0.9);

    agent.setAlpha(0.001);
    //agent.setBatchSize(10);

    vector<int> sensors;
    Vector state0, state1;
    int action0, action1;
    double reward = 0;
    double epsilon = 1;
    int epochs = 1000;

    int wins = 0, loses = 0;

    FILE* pFile = fopen("application.log", "w");
    Output2FILE::Stream() = pFile;
    FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

    for (int e = 0; e < epochs; e++) {
        cout << "Epoch " << e << endl;

        task.getEnvironment()->reset();

        while(!task.isFinished()) {
            //cout << maze->toString() << endl;

            sensors = maze->getSensors();
            state0 = encodeState(&sensors);
            network.activate(&state0);
            action0 = chooseAction(network.getOutput(), epsilon);
            maze->performAction(action0);

            sensors = maze->getSensors();
            state1 = encodeState(&sensors);
            network.activate(&state1);
            action1 = chooseAction(network.getOutput(), epsilon);

            reward = task.getReward();
            agent.train(&state0, action0, &state1, action1, reward);
        }

        if (reward > 0) {
            wins++;
        }
        else {
            loses++;
        }

        //cout << maze->toString() << endl;
        cout << wins << " / " << loses << endl;
        FILE_LOG(logDEBUG1) << wins << " " << loses;


        if (epsilon > 0.1) {
            epsilon -= (1.0 / epochs);
        }
    }
}

void sampleAC() {
    MazeTask task;
    Maze* maze = task.getEnvironment();

    NeuralNetwork nc;

    nc.addLayer("input", 64, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    nc.addLayer("hidden0", 164, NeuralGroup::RELU, NeuralNetwork::HIDDEN);
    nc.addLayer("hidden1", 150, NeuralGroup::RELU, NeuralNetwork::HIDDEN);
    nc.addLayer("output", 4, NeuralGroup::TANH, NeuralNetwork::OUTPUT);

    // feed-forward connections
    nc.addConnection("input", "hidden0");
    nc.addConnection("hidden0", "hidden1");
    nc.addConnection("hidden1", "output");

    //ADAM optimizer_c(&nc);
    BackProp optimizer_c(&nc, 1e-6, 0.9, true);
    QLearning critic(&optimizer_c, &nc, 0.9, 0);
    critic.setAlpha(0.1);

    NeuralNetwork na;

    na.addLayer("input", 64, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    na.addLayer("hidden0", 164, NeuralGroup::RELU, NeuralNetwork::HIDDEN);
    na.addLayer("hidden1", 150, NeuralGroup::RELU, NeuralNetwork::HIDDEN);
    na.addLayer("output", 4, NeuralGroup::LINEAR, NeuralNetwork::OUTPUT);

    // feed-forward connections
    na.addConnection("input", "hidden0");
    na.addConnection("hidden0", "hidden1");
    na.addConnection("hidden1", "output");

    //ADAM optimizer_a(&na);
    BackProp optimizer_a(&nc, 1e-6, 0.9, true);
    ActorLearning actor(&optimizer_a, &na);
    actor.setAlpha(0.1);


    vector<int> sensors;
    Vector state0, state1;
    int action0;
    double reward = 0;
    double value0, value1;
    double epsilon = 1;
    int epochs = 1000;

    int wins = 0, loses = 0;

    FILE* pFile = fopen("application.log", "w");
    Output2FILE::Stream() = pFile;
    FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

    for (int e = 0; e < epochs; e++) {
        cout << "Epoch " << e << endl;

        task.getEnvironment()->reset();

        while(!task.isFinished()) {
            //cout << maze->toString() << endl;

            sensors = maze->getSensors();
            state0 = encodeState(&sensors);
            nc.activate(&state0);
            na.activate(&state0);
            action0 = chooseAction(na.getOutput(), epsilon);
            value0 = (*nc.getOutput())[action0];
            maze->performAction(action0);

            sensors = maze->getSensors();
            state1 = encodeState(&sensors);
            nc.activate(&state1);
            value1 = (*nc.getOutput())[action0];
            reward = task.getReward();

            critic.train(&state0, action0, &state1, reward);
            actor.train(&state0, action0, value0, value1, reward);
        }

        if (reward > 0) {
            wins++;
        }
        else {
            loses++;
        }

        //cout << maze->toString() << endl;
        cout << wins << " / " << loses << endl;
        FILE_LOG(logDEBUG1) << wins << " " << loses;


        if (epsilon > 0.1) {
            epsilon -= (1.0 / epochs);
        }
    }
}