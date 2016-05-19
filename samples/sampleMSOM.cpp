//
// Created by mpechac on 16. 5. 2016.
//

#include <iostream>
#include "samples.h"
#include "../dataset/Dataset.h"
#include "../network/Define.h"
#include "../network/som/MSOM.h"

using namespace NeuroNet;

void sampleMSOM() {
    Dataset dataset;
    DatasetConfig config = {1, 0, "", 0};
    dataset.load("../data/mg30.dat", config);
    dataset.normalize();

    MSOM msom(1, 8, 8, SIGMOID);
    double epochs = 300;
    msom.initTraining(0.01, 0.01, 0.5, 0.5, epochs);

    for(int t = 0; t < epochs; t++) {
        for(int i = 0; i < dataset.getData()->size(); i++) {
            msom.train(dataset.getData()->at(i).first.data());
        }
        cout << "qError: " << msom.getError() << " WD: " << msom.getWinnerDifferentiation() << endl;
        msom.paramDecay();
    }
}