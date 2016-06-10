#include <iostream>
#include "../network/NeuralNetwork.h"
#include "../network/Define.h"
#include "../algorithm/BackProp.h"

using namespace NeuroNet;

void sampleBP() {
    double trainingSet[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double targetSet[4][1] = {{0},{1},{1},{0}};
    double mse = 1;

    NeuralNetwork network;
    
    network.addLayer("input", 2, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    network.addLayer("biasH", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    network.addLayer("biasO", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    network.addLayer("hidden", 4, NeuralGroup::SIGMOID, NeuralNetwork::HIDDEN);
    network.addLayer("output", 1, NeuralGroup::SIGMOID, NeuralNetwork::OUTPUT);

    // feed-forward connections
    network.addConnection("input", "hidden");
    network.addConnection("hidden", "output");
    // bias connections
    network.addConnection("biasH", "hidden");
    network.addConnection("biasO", "output");

    BackProp bp(&network);
    bp.setAlpha(0.01);
    //bp.setWeightDecay(0.001);

    while(mse > 0.01) {
      mse = 0;
      for(int i = 0; i < 4; i++) {
        mse += bp.train(trainingSet[i], targetSet[i]);
        //cout << network.getScalarOutput() << endl;
      }
      cout << "Error " << mse << endl;
    }

    for(int i = 0; i < 4; i++) {
        network.setInput(trainingSet[i]);
        network.onLoop();
        cout << (*network.getOutput()) << endl;
    }

    //cout << mse << endl;
    cout << "Uspesne ukoncene." << endl;
}