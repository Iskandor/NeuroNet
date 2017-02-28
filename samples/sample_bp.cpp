#include <iostream>
#include "../network/NeuralNetwork.h"
#include "../network/Define.h"
#include "../algorithm/BackProp.h"
#include "../log/log.h"

using namespace NeuroNet;

void sampleBP() {
    double trainingSet[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double targetSet[4][1] = {{0},{1},{1},{0}};
    double mse = 1;

    NeuralNetwork network;
    
    network.addLayer("input", 2, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    network.addLayer("hidden0", 4, NeuralGroup::SIGMOID, NeuralNetwork::HIDDEN);
    network.addLayer("output", 1, NeuralGroup::SIGMOID, NeuralNetwork::OUTPUT);

    // feed-forward connections
    network.addConnection("input", "hidden0");
    network.addConnection("hidden0", "output");

    BackProp bp(&network, 0.1e-6, 0.9);
    bp.setAlpha(0.1);

    FILE* pFile = fopen("application.log", "w");
    Output2FILE::Stream() = pFile;
    FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");


    while(mse > 0.01) {
      mse = 0;
      for(int i = 0; i < 4; i++) {
        mse += bp.train(trainingSet[i], targetSet[i]);
        //cout << network.getScalarOutput() << endl;
      }
      cout << "Error " << mse << endl;
      FILE_LOG(logDEBUG1) << mse;
    }

    for(int i = 0; i < 4; i++) {
        network.setInput(trainingSet[i]);
        network.onLoop();
        cout << (*network.getOutput()) << endl;
    }

    //cout << mse << endl;
    cout << "Uspesne ukoncene." << endl;
}