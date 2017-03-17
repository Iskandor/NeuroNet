//
// Created by mpechac on 10. 3. 2016.
//

#include <c++/iostream>
#include "../network/som/SOM.h"
#include "../dataset/Dataset.h"
#include "../network/Define.h"

using namespace NeuroNet;

void sampleSOM() {
    Dataset dataset;
    DatasetConfig config = {13, 1, ",", 13};
    dataset.load("../data/wine.dat", config);
    dataset.normalize();

    SOM somNetwork(4, 8, 8, NeuralGroup::SIGMOID);
    double epochs = 1000;
    somNetwork.initTraining(0.01, epochs);

    for(int t = 0; t < epochs; t++) {
        dataset.permute();
        for(int i = 0; i < dataset.getData()->size(); i++) {
            somNetwork.train(&dataset.getData()->at(i).first);
        }
        cout << "qError: " << somNetwork.getError() << " WD: " << somNetwork.getWinnerDifferentiation() << endl;
        somNetwork.paramDecay();
    }
}