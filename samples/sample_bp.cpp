#include <iostream>
#include "../network/NeuralNetwork.h"
#include "../algorithm/optimizer/ADAM.h"
#include "../algorithm/optimizer/BackProp.h"
#include "../algorithm/optimizer/RMSProp.h"


using namespace NeuroNet;

void sampleBP() {
    double mse = 1;

    Vector*    training[4];
    Vector*    target[4];

    for(int i = 0; i < 4; i++) {
        training[i] = new Vector(2);
        target[i] = new Vector(1);
    }

    *training[0] = Vector(2, {0,0});
    *training[1] = Vector(2, {0,1});
    *training[2] = Vector(2, {1,0});
    *training[3] = Vector(2, {1,1});

    *target[0] = Vector(1, {0});
    *target[1] = Vector(1, {1});
    *target[2] = Vector(1, {1});
    *target[3] = Vector(1, {0});

    NeuralNetwork network;
    
    network.addLayer("input", 2, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    network.addLayer("hidden0", 4, NeuralGroup::SIGMOID, NeuralNetwork::HIDDEN);
    network.addLayer("output", 1, NeuralGroup::SIGMOID, NeuralNetwork::OUTPUT);

    // feed-forward connections
    network.addConnection("input", "hidden0");
    network.addConnection("hidden0", "output");

    //BackProp bp(&network, 1e-6, 0.9, true);
    RMSProp bp(&network);
    //ADAM bp(&network);
    bp.setAlpha(0.1);
    //bp.setBatchSize(4);

    for(int e = 0; e < 3000; e++) {
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
}