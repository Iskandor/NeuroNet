#include <iostream>
#include "NeuralNetwork.h"
#include "Define.h"
#include <TDLambda.h>
#include <NBandit.h>
#include <BanditGame.h>

void sampleTD() {
    double sumReward = 0;
    int time = 0;

    NeuralNetwork network;
    
    NeuralGroup* inputGroup = network.addLayer(5, IDENTITY, NeuralNetwork::INPUT);
    NeuralGroup* biasUnit = network.addLayer(1, BIAS, NeuralNetwork::HIDDEN);
    NeuralGroup* hiddenGroup = network.addLayer(100, SIGMOID, NeuralNetwork::HIDDEN);
    NeuralGroup* outputGroup = network.addLayer(1, SIGMOID, NeuralNetwork::OUTPUT);

    // feed-forward connections
    network.addConnection(inputGroup, hiddenGroup);
    network.addConnection(hiddenGroup, outputGroup);
    // bias connections
    network.addConnection(biasUnit, hiddenGroup);
    network.addConnection(biasUnit, outputGroup);

    TDLambda td(&network, 0.9, 0.9);
    td.setAlpha(0.01);

    BanditGame game(5, 10);
    double epsilon = 0.01;

    vectorN<double> *action = new vectorN<double>(10);
    vectorN<double> *state = new vectorN<double>(5);
    double tmpReward = 0;

    while(true) {
      double maxOutput = -1;
      int action_i = 0;
      double reward = 0;

      for (int i = 0; i < action->size(); i++) {
        action->set(0);
        state->set(0);
        action->set(i, 1);

        game.evaluateAction(action, state);
        network.setInput(state);
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

      action->set(0);
      action->set(action_i, 1);    

      game.updateState(action);
      reward = game.getReward();
      sumReward += reward;

      // 3. update
      if (time > 0) {
        td.train(game.getState()->getVector(), &reward);
        cout << time << " " << game.getIndex() << " " << reward << " " << action_i << " " <<  network.getScalarOutput()  << " " << game.getBandit(game.getIndex())->getProbability(action_i) << " " << endl;
      }
      time++;    

      // 4. check whether terminal state was reached
      if (time > 10000) {
        cout << "10000! Reward:" << sumReward << endl; 
        time = 0;
        sumReward = 0;
      }
    }

    cout << "Uspesne ukoncene." << endl;
}
