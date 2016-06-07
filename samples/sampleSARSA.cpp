//
// Created by user on 25. 2. 2016.
//

#include <iostream>
#include "../network/NeuralNetwork.h"
#include "../network/Define.h"
#include "../algorithm/rl/SARSA.h"
#include "../algorithm/GreedyPolicy.h"
#include "../log/log.h"
#include "Maze.h"

using namespace NeuroNet;

void sampleSARSA() {
  double sumReward = 0;
  const int dim = 3;
  int time = 0;

  NeuralNetwork network;

  NeuralGroup* inputGroup = network.addLayer("input", 4+dim*dim, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
  NeuralGroup* biasUnitH = network.addLayer("biasH", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
  NeuralGroup* biasUnitO = network.addLayer("biasO", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
  NeuralGroup* hiddenGroup = network.addLayer("hidden", 5, NeuralGroup::SIGMOID, NeuralNetwork::HIDDEN);
  NeuralGroup* outputGroup = network.addLayer("output", 1, NeuralGroup::TANH, NeuralNetwork::OUTPUT);


  // feed-forward connections
  network.addConnection(inputGroup, hiddenGroup);
  network.addConnection(hiddenGroup, outputGroup);
  // bias connections
  network.addConnection(biasUnitH, hiddenGroup);
  network.addConnection(biasUnitO, outputGroup);

  SARSA agent(&network, 0.9, 0.9);
  agent.setAlpha(0.1);

  Maze maze(dim);
  maze.reset();

  GreedyPolicy policy(&network, &maze);
  policy.setEpsilon(0.01);

  VectorXd action0(4);
  VectorXd action1(4);
  VectorXd state0(dim*dim);
  VectorXd state1(dim*dim);

  int episode = 0;

  FILE* pFile = fopen("application.log", "w");
  Output2FILE::Stream() = pFile;
  FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

  while(episode < 2000) {
    double reward = 0;

    state0 = *maze.getState();
    policy.getActionQ(&state0, &action0);
    maze.updateState(&action0);
    state1 = *maze.getState();
    reward = maze.getReward();
    sumReward += reward;
    policy.getActionQ(&state1, &action1);

    //cout << time << " a = " << action_i << " r = " << reward << " Q(s,a) = " << (*network.getOutput())[action_i] << endl;

    // 3. update
    agent.train(&state0, &action0, &state1, &action1, reward);
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
