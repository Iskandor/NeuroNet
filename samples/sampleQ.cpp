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
#include "../algorithm/GreedyPolicy.h"
#include "../algorithm/rl/QNatLearning.h"
#include "MazeTask.h"

using namespace NeuroNet;

/*
void sampleQ() {
  double sumReward = 0;
  int time = 0;
  int dim = 3;

  NeuralNetwork network;

  NeuralGroup* inputGroup = network.addLayer("input", 4+dim*dim, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
  NeuralGroup* biasUnitH = network.addLayer("biasH", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
  NeuralGroup* biasUnitO = network.addLayer("biasO", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
  NeuralGroup* hiddenGroup = network.addLayer("hidden", 32, NeuralGroup::TANH, NeuralNetwork::HIDDEN);
  NeuralGroup* outputGroup = network.addLayer("output", 1, NeuralGroup::TANH, NeuralNetwork::OUTPUT);

  // feed-forward connections
  network.addConnection(inputGroup, hiddenGroup);
  network.addConnection(hiddenGroup, outputGroup);
  // bias connections
  network.addConnection(biasUnitH, hiddenGroup);
  network.addConnection(biasUnitO, outputGroup);

  QLearning qAgent(&network, 0.9, 0.9);
  qAgent.setAlpha(0.00001);

  MazeOld maze(dim);
  maze.reset();

  VectorXd action(4);
  VectorXd state0(dim*dim);
  VectorXd state1(dim*dim);

  int episode = 0;
  GreedyPolicy policy(&network, &maze);
  policy.setEpsilon(0);

  FILE* pFile = fopen("application.log", "w");
  Output2FILE::Stream() = pFile;
  FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

  while(episode < 2000) {
    double reward = 0;

    state0 = *maze.getState();
    policy.getActionQ(&state0, &action);
    maze.updateState(&action);
    state1 = *maze.getState();
    reward = maze.getReward();
    sumReward += reward;

    //cout << time << " a = " << action_i << " r = " << reward << " Q(s,a) = " << (*network.getOutput())[action_i] << endl;

    // 3. update
    qAgent.train(&state0, &action, &state1, reward);
    time++;

    // 4. check whether terminal state was reached
    if (maze.isFinished()) {
      cout << "Finished episode " << episode << "! " << time << " Reward:" << sumReward << endl;
      FILE_LOG(logDEBUG1) << sumReward;

      time = 0;
      sumReward = 0;
      maze.reset();
      episode++;
    }
  }

  NetworkUtils::saveNetwork("qmaze.net", &network);

  cout << "Uspesne ukoncene." << endl;
}
*/

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

    QLearning agent(&network, 0.9, 0, 1e-6, 0.9);
    agent.setAlpha(0.1);
    //agent.setBatchSize(10);

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