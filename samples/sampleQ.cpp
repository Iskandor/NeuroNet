//
// Created by user on 25. 2. 2016.
//

#include <iostream>
#include "../network/NeuralNetwork.h"
#include "../network/Define.h"
#include "../algorithm/QLearning.h"
#include "Maze.h"
#include "../log/log.h"

void sampleQ() {
  double sumReward = 0;
  int time = 0;
  int dim = 3;

  NeuralNetwork network;

  NeuralGroup* inputGroup = network.addLayer("input", 4+dim*dim, IDENTITY, NeuralNetwork::INPUT);
  NeuralGroup* biasUnitH = network.addLayer("biasH", 1, BIAS, NeuralNetwork::HIDDEN);
  NeuralGroup* biasUnitO = network.addLayer("biasO", 1, BIAS, NeuralNetwork::HIDDEN);
  NeuralGroup* hiddenGroup = network.addLayer("hidden", 25, SIGMOID, NeuralNetwork::HIDDEN);
  NeuralGroup* outputGroup = network.addLayer("output", 1, IDENTITY, NeuralNetwork::OUTPUT);


  // feed-forward connections
  network.addConnection(inputGroup, hiddenGroup);
  network.addConnection(hiddenGroup, outputGroup);
  // bias connections
  network.addConnection(biasUnitH, hiddenGroup);
  network.addConnection(biasUnitO, outputGroup);

  QLearning qAgent(&network, 0.9, 0.9);
  qAgent.setAlpha(0.1);

  Maze maze(dim);
  maze.reset();
  double epsilon = 0.01;

  VectorXd action(4);
  VectorXd state0(dim*dim);
  VectorXd state1(dim*dim);

  int episode = 0;

  FILE* pFile = fopen("application.log", "w");
  Output2FILE::Stream() = pFile;
  FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

  while(episode < 1000) {
    double maxOutput = -INFINITY;
    int action_i = 0;
    double reward = 0;

    state0 = *maze.getState();

    for (int i = 0; i < action.size(); i++) {
        action.fill(0);
        action[i] = 1;

        double roll = rand() % 100;
        if (roll < epsilon * 100) {
          action_i = i;
          break;
        }

        VectorXd input(state0.size() + action.size());
        input << state0, action;
        network.activate(&input);

        if (maxOutput < network.getScalarOutput()) {
          action_i = i;
          maxOutput = network.getScalarOutput();
        }

        //cout << "a = " << i << " Q(s,a) = " << (*network.getOutput())[i] << endl;
    }

    action.fill(0);
    action[action_i] = 1;

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

  cout << "Uspesne ukoncene." << endl;
}
