//
// Created by mpechac on 10. 3. 2016.
//

#include <c++/iostream>
#include "sampleSOM.h"
#include "../network/som/SOM.h"
#include "../dataset/Dataset.h"
#include "../network/Define.h"

void sampleSOM() {
    Dataset dataset;
    DatasetConfig config = {4, 1, ",", 4};
    dataset.load("../data/hayes-roth.dat", config);
    dataset.normalize();

    SOM somNetwork(4, 8, 8, SIGMOID);
    somNetwork.reset(0.01);

    for(int t = 0; t < 10; t++) {
        for(int i = 0; i < dataset.getData()->size(); i++) {
            somNetwork.train(dataset.getData()->at(i).data());
        }
        cout << somNetwork.getError()  << endl;
        somNetwork.paramDecay();
    }
}