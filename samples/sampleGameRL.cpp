//
// Created by user on 28. 5. 2017.
//

#include <backend/flab/RandomGenerator.h>
#include <log/log.h>
#include <network/NeuralNetwork.h>
#include <algorithm/optimizer/ADAM.h>
#include <algorithm/rl/QLearning.h>
#include "sampleGameRL.h"
#include "TicTacToeTask.h"

using namespace NeuroNet;

sampleGameRL::sampleGameRL() {

}

sampleGameRL::~sampleGameRL() {

}


void sampleGameRL::sampleTicTacToe() {
    TicTacToeTask task;
    TicTacToe *game = task.getEnvironment();

    NeuralNetwork network;

    network.addLayer("input", 10, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    network.addLayer("hidden0", 81, NeuralGroup::RELU, NeuralNetwork::HIDDEN);
    network.addLayer("hidden1", 18, NeuralGroup::RELU, NeuralNetwork::HIDDEN);
    network.addLayer("output", 9, NeuralGroup::LINEAR, NeuralNetwork::OUTPUT);

    // feed-forward connections
    network.addConnection("input", "hidden0");
    network.addConnection("hidden0", "hidden1");
    network.addConnection("hidden1", "output");

    ADAM optimizer(&network);
    //BackProp optimizer(&network, 1e-6, 0.9, true, GradientDescent::NATURAL);
    QLearning agent(&optimizer, &network, 0.9, 0);
    agent.setAlpha(0.001);

    Player p1;
    p1.playerID = 1;
    p1.agent = &agent;

    Player p2;
    p2.playerID = -1;
    p2.agent = &agent;

    vector<double> sensors;
    Vector state0, state1;
    int action;
    double reward = 0;
    double epsilon = 1;
    int epochs = 10000;
    int player;
    int round;

    int wins = 0, loses = 0;

    FILE* pFile = fopen("application.log", "w");
    Output2FILE::Stream() = pFile;
    FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

    for (int e = 0; e < epochs; e++) {
        cout << "Epoch " << e << endl;

        task.getEnvironment()->reset();
        round = 0;

        while(!task.isFinished()) {
            round++;
            //cout << game->toString() << endl;

            player = game->activePlayer();

            sensors = game->getSensors();
            state0 = encodeState(player, &sensors);
            network.activate(&state0);
            action = chooseAction(network.getOutput(), epsilon);
            game->performAction(action);

            sensors = game->getSensors();
            state1 = encodeState(player, &sensors);
            reward = task.getReward(player);
            agent.train(&state0, action, &state1, reward);
        }

        //cout << game->toString() << endl;

        if (task.Winner() == p1.playerID) {
            wins++;
        }
        if (task.Winner() == p2.playerID) {
            loses++;
        }

        cout << wins << " / " << loses << " " << round << endl;

        FILE_LOG(logDEBUG1) << wins << " " << loses;


        if (epsilon > 0.1) {
            epsilon -= (1.0 / epochs);
        }
    }
}

Vector sampleGameRL::encodeState(int p_playerID, vector<double> *p_sensors) {
    Vector res(10);

    for(int i = 0; i < p_sensors->size(); i++) {
        res[i] = (*p_sensors)[i];
    }
    res[9] = p_playerID;

    return Vector(res);
}

int sampleGameRL::chooseAction(Vector *p_input, double epsilon) {
    int action = 0;
    double random = RandomGenerator::getInstance().random();

    if (random < epsilon) {
        action = RandomGenerator::getInstance().random(0, 8);
    }
    else {
        action = p_input->maxIndex();
    }

    return action;
}
