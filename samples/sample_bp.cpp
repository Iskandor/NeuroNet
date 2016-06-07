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
    
    NeuralGroup* inputGroup = network.addLayer("input", 2, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    NeuralGroup* biasUnitH = network.addLayer("biasH", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    NeuralGroup* biasUnitO = network.addLayer("biasO", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    NeuralGroup* hiddenGroup = network.addLayer("hidden", 4, NeuralGroup::SIGMOID, NeuralNetwork::HIDDEN);
    NeuralGroup* outputGroup = network.addLayer("output", 1, NeuralGroup::SIGMOID, NeuralNetwork::OUTPUT);

    // feed-forward connections
    network.addConnection(inputGroup, hiddenGroup);
    network.addConnection(hiddenGroup, outputGroup);
    // bias connections
    network.addConnection(biasUnitH, hiddenGroup);
    network.addConnection(biasUnitO, outputGroup);

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