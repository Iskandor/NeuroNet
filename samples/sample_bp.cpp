#include <iostream>
#include "../network/NeuralNetwork.h"
#include "../network/Define.h"
#include "../algorithm/BackProp.h"
#include "../log/log.h"
#include "../network/RandomGenerator.h"
#include "../algorithm/RMSProp.h"

using namespace NeuroNet;

void sampleBP() {
    double mse = 1;

    VectorXd*    training[4];
    VectorXd*    target[4];

    for(int i = 0; i < 4; i++) {
        training[i] = new VectorXd(2);
        target[i] = new VectorXd(1);
    }

    *training[0] << 0,0;
    *training[1] << 0,1;
    *training[2] << 1,0;
    *training[3] << 1,1;

    *target[0] << 0;
    *target[1] << 1;
    *target[2] << 1;
    *target[3] << 0;

    NeuralNetwork network;
    
    network.addLayer("input", 2, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    network.addLayer("hidden0", 4, NeuralGroup::SIGMOID, NeuralNetwork::HIDDEN);
    network.addLayer("output", 1, NeuralGroup::SIGMOID, NeuralNetwork::OUTPUT);

    // feed-forward connections
    network.addConnection("input", "hidden0");
    network.addConnection("hidden0", "output");

    BackProp bp(&network, GradientDescent::REGULAR, 1e-6, 0.99);
    bp.setAlpha(0.1);
    //bp.setBatchSize(4);

    while(mse > 0.01) {
      mse = 0;
      for(int i = 0; i < 4; i++) {
        mse += bp.train(training[i], target[i]);
        //cout << network.getScalarOutput() << endl;
      }
      cout << "Error " << mse << endl;
    }

    for(int i = 0; i < 4; i++) {
        network.activate(training[i]);
        cout << (*network.getOutput()) << endl;
    }

    //cout << mse << endl;
    cout << "Uspesne ukoncene." << endl;
}