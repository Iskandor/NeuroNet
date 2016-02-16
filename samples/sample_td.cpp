#include <iostream>
#include <stdlib.h>
#include "../network/NeuralNetwork.h"
#include "../network/Define.h"
#include "../algorithm/TDLambda.h"
#include "Maze.h"

void sampleTD() {
    double sumReward = 0;
    int time = 0;
    int dim = 4;

    NeuralNetwork network;
    
    NeuralGroup* inputGroup = network.addLayer(dim*dim, IDENTITY, NeuralNetwork::INPUT);
    NeuralGroup* biasUnit = network.addLayer(1, BIAS, NeuralNetwork::HIDDEN);
    NeuralGroup* hiddenGroup = network.addLayer(32, SIGMOID, NeuralNetwork::HIDDEN);
    NeuralGroup* outputGroup = network.addLayer(1, SIGMOID, NeuralNetwork::OUTPUT);

    // feed-forward connections
    network.addConnection(inputGroup, hiddenGroup);
    network.addConnection(hiddenGroup, outputGroup);
    // bias connections
    network.addConnection(biasUnit, hiddenGroup);
    network.addConnection(biasUnit, outputGroup);

    TDLambda td(&network, 0.9, 0.9);
    td.setAlpha(0.01);

    Maze maze(dim);
    maze.reset();
    double epsilon = 0.01;    

    VectorXd action(4);
    VectorXd state(dim*dim);

    while(true) {
      double maxOutput = -1;
      int action_i = 0;
      double reward = 0;

      for (int i = 0; i < action.size(); i++) {
        action.Zero(action.size());
        action[i] = 1;

        maze.evaluateAction(&action, &state);
        network.setInput(&state);
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

      action.Zero(action.size());
      action[action_i] = 1;

      maze.updateState(&action);
      reward = maze.getReward();
      sumReward += reward;

      // 3. update
      if (time > 0) {
        td.train(maze.getState()->data(), &reward);
      }
      time++;


      /*
      cout << time << " " << reward << " " << action_i << " " <<  network.getScalarOutput() << endl;
      for(auto i = 0; i < dim; i++) {
        for(auto j = 0; j < dim; j++) {
          cout << maze.getState()->at(i*dim + j);
        }
        cout << endl;
      }
       */

      // 4. check whether terminal state was reached
      if (time > 10000 || maze.isFinished()) {
        cout << "Finish! " << time << " Reward:" << sumReward << endl;
        time = 0;
        sumReward = 0;
        maze.reset();
      }
    }

    cout << "Uspesne ukoncene." << endl;
}
