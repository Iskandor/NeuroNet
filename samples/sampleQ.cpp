//
// Created by user on 25. 2. 2016.
//

#include <iostream>
#include "../network/NeuralNetwork.h"
#include "../network/Define.h"
#include "Maze.h"
#include "../log/log.h"
#include "../algorithm/rl/QLearning.h"
#include "../network/NetworkUtils.h"
#include "../network/RandomGenerator.h"
#include "../algorithm/GreedyPolicy.h"

using namespace NeuroNet;

void sampleQ() {
  double sumReward = 0;
  int time = 0;
  int dim = 3;

  NeuralNetwork network;

  NeuralGroup* inputGroup = network.addLayer("input", 4+dim*dim, IDENTITY, NeuralNetwork::INPUT);
  NeuralGroup* biasUnitH = network.addLayer("biasH", 1, BIAS, NeuralNetwork::HIDDEN);
  NeuralGroup* biasUnitO = network.addLayer("biasO", 1, BIAS, NeuralNetwork::HIDDEN);
  NeuralGroup* hiddenGroup = network.addLayer("hidden", 3, SIGMOID, NeuralNetwork::HIDDEN);
  NeuralGroup* outputGroup = network.addLayer("output", 1, TANH, NeuralNetwork::OUTPUT);

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

  VectorXd action(4);
  VectorXd state0(dim*dim);
  VectorXd state1(dim*dim);

  int episode = 0;
  GreedyPolicy policy(&network, &maze);
  policy.setEpsilon(0.01);

  FILE* pFile = fopen("application.log", "w");
  Output2FILE::Stream() = pFile;
  FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

  while(episode < 6000) {
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
