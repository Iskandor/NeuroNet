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

  NeuralGroup* inputGroup = network.addLayer("input", dim*dim+4, IDENTITY, NeuralNetwork::INPUT);
  NeuralGroup* biasUnitH = network.addLayer("biasH", 1, BIAS, NeuralNetwork::HIDDEN);
  NeuralGroup* biasUnitO = network.addLayer("biasO", 1, BIAS, NeuralNetwork::HIDDEN);
  NeuralGroup* hiddenGroup = network.addLayer("hidden", 9, SIGMOID, NeuralNetwork::HIDDEN);
  NeuralGroup* outputGroup = network.addLayer("output", 1, TANH, NeuralNetwork::OUTPUT);


  // feed-forward connections
  network.addConnection(inputGroup, hiddenGroup);
  network.addConnection(hiddenGroup, outputGroup);
  // bias connections
  network.addConnection(biasUnitH, hiddenGroup);
  network.addConnection(biasUnitO, outputGroup);

  QLearning qAgent(&network, 0.99);
  qAgent.setAlpha(.01);

  Maze maze(dim);
  maze.reset();
  double epsilon = 0.01;

  VectorXd action(4);
  VectorXd state0(dim*dim);
  VectorXd state1(dim*dim);

  for(int episode = 0; episode < 1000000; episode++) {
    double maxOutput = -1;
    int action_i = 0;
    double reward = 0;

    state0 = *maze.getState();
/*
    for(auto i = 0; i < dim; i++) {
      for(auto j = 0; j < dim; j++) {
        cout << (*maze.getState())(i*dim + j);
      }
      cout << endl;
    }
    */

    for (int i = 0; i < action.size(); i++) {

      action.fill(0);
      action[i] = 1;

      maze.evaluateAction(&action, &state1);
      VectorXd input(state1.size() + action.size());
      input << state1, action;
      network.setInput(&input);
      network.onLoop();

      double roll = static_cast<double>(rand()) / RAND_MAX;

      if (roll < epsilon) {
        action_i = i;
        break;
      }

      if (maxOutput < network.getScalarOutput())
      {
        action_i = i;
        maxOutput = network.getScalarOutput();
      }
    }

    action.fill(0);
    action[action_i] = 1;

    maze.updateState(&action);
    state1 = *maze.getState();
    reward = maze.getReward();
    sumReward += reward;

    // 3. update
    qAgent.train(&state0, &action, &state1, reward);
    time++;


    cout << time << " " << reward << " " << action_i << " " <<  network.getScalarOutput() << endl;

    // 4. check whether terminal state was reached
    if (time > 10000 || maze.isFinished()) {
      cout << "Finish! " << time << " Reward:" << sumReward << endl;
      //cout << time << " " << reward << " " << action_i << " " <<  network.getScalarOutput() << endl;
      /*
      for(auto i = 0; i < dim; i++) {
        for(auto j = 0; j < dim; j++) {
          cout << (*maze.getState())(i*dim + j);
        }
        cout << endl;
      }
       */

      time = 0;
      sumReward = 0;
      maze.reset();
    }
  }

  cout << "Uspesne ukoncene." << endl;
}
