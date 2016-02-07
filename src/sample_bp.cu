#include <iostream>
#include "NeuralNetwork.h"
#include "BackProp.h"
#include "Define.h"

void sampleBP() {
    double trainingSet[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double targetSet[4][1] = {{0},{0},{0},{1}};
    double mse = 1;

    NeuralNetwork network;
    
    NeuralGroup* inputGroup = network.addLayer(2, IDENTITY, NeuralNetwork::INPUT);
    NeuralGroup* biasUnit = network.addLayer(1, BIAS, NeuralNetwork::HIDDEN);
    NeuralGroup* hiddenGroup = network.addLayer(2, SIGMOID, NeuralNetwork::HIDDEN);
    NeuralGroup* outputGroup = network.addLayer(1, SIGMOID, NeuralNetwork::OUTPUT);

    // feed-forward connections
    network.addConnection(inputGroup, hiddenGroup);
    network.addConnection(hiddenGroup, outputGroup);
    // bias connections
    network.addConnection(biasUnit, hiddenGroup);
    network.addConnection(biasUnit, outputGroup);

    network.init();

    BackProp bp(&network);
    bp.setAlpha(0.5);
    //bp.setWeightDecay(0.001);

    while(mse > 0.01) {
      mse = 0;
      for(int i = 0; i < 4; i++) {
        mse += bp.train(trainingSet[i], targetSet[i]);
        //cout << network.getOutput()[0] << endl;        
      }
      cout << "Error " << mse << endl;
    }

    for(int i = 0; i < 4; i++) {
      network.setInput(trainingSet[i]);
      network.onLoop();
      cout << network.getOutput()[0] << endl;        
    }

    //cout << mse << endl;
    cout << "Uspesne ukoncene." << endl;
}