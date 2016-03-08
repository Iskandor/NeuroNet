//
// Created by user on 25. 2. 2016.
//

#include <iostream>
#include "../network/NeuralNetwork.h"
#include "../network/Define.h"
#include "../algorithm/QLearning.h"
#include "Maze.h"

void sampleQ() {
  double sumReward = 0;
  int time = 0;
  int dim = 3;

  NeuralNetwork network;

  NeuralGroup* inputGroup = network.addLayer("input", dim*dim, IDENTITY, NeuralNetwork::INPUT);
  NeuralGroup* biasUnitH = network.addLayer("biasH", 1, BIAS, NeuralNetwork::HIDDEN);
  NeuralGroup* biasUnitO = network.addLayer("biasO", 1, BIAS, NeuralNetwork::HIDDEN);
  NeuralGroup* hiddenGroup = network.addLayer("hidden", 25, TANH, NeuralNetwork::HIDDEN);
  NeuralGroup* outputGroup = network.addLayer("output", 4, TANH, NeuralNetwork::OUTPUT);


  // feed-forward connections
  network.addConnection(inputGroup, hiddenGroup);
  network.addConnection(hiddenGroup, outputGroup);
  // bias connections
  network.addConnection(biasUnitH, hiddenGroup);
  network.addConnection(biasUnitO, outputGroup);

  QLearning qAgent(&network, 0.99, 0.99);
  qAgent.setAlpha(.01);

  Maze maze(dim);
  maze.reset();
  double epsilon = 0.5;

  VectorXd action(4);
  VectorXd state0(dim*dim);
  VectorXd state1(dim*dim);

  for(int episode = 0; episode < 1000000; episode++) {
    double maxOutput = -1;
    int action_i = 0;
    double reward = 0;

    state0 = *maze.getState();

    for(auto i = 0; i < dim; i++) {
      for(auto j = 0; j < dim; j++) {
        cout << (*maze.getState())(i*dim + j);
      }
      cout << endl;
    }

    network.setInput(&state0);
    network.onLoop();

    double roll = rand() % 100;

    if (roll < epsilon * 100) {
      action_i = rand() % 4;

    }
    else {
      for (int i = 0; i < action.size(); i++) {
        action.fill(0);
        action[i] = 1;

        if (maxOutput < (*network.getOutput())[i])
        {
          action_i = i;
          maxOutput = (*network.getOutput())[i];
        }

        cout << "a = " << i << " Q(s,a) = " << (*network.getOutput())[i] << endl;
      }
    }

    action.fill(0);
    action[action_i] = 1;

    maze.updateState(&action);
    state1 = *maze.getState();
    reward = maze.getReward();
    sumReward += reward;

    cout << time << " a = " << action_i << " r = " << reward << " Q(s,a) = " << (*network.getOutput())[action_i] << endl;

    // 3. update
    qAgent.train(&state0, &action, &state1, reward);
    time++;

    // 4. check whether terminal state was reached
    if (time > 10000 || maze.isFinished()) {
      cout << "Finish! " << time << " Reward:" << sumReward << endl;
      //cout << time << " " << reward << " " << action_i << " " <<  network.getScalarOutput() << endl;

      for(auto i = 0; i < dim; i++) {
        for(auto j = 0; j < dim; j++) {
          cout << (*maze.getState())(i*dim + j);
        }
        cout << endl;
      }

      time = 0;
      sumReward = 0;
      epsilon *= 0.999;
      maze.reset();
    }
  }

  cout << "Uspesne ukoncene." << endl;
}
