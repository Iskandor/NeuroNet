//
// Created by mpechac on 20. 5. 2016.
//

#include <iostream>
#include "../network/Define.h"
#include "../network/som/MSOM.h"
#include "../network/RandomGenerator.h"
#include "../network/NetworkUtils.h"
#include "../network/som/RecSOM.h"

using namespace NeuroNet;

void sampleDigits() {
    RandomGenerator generator;
    RecSOM msom(10, 7, 7, EXPONENTIAL);
    double epochs = 300;
    msom.initTraining(0.01, 0.01, 1.1, 1, epochs);

    int length = 0;
    int digit;
    VectorXd binDigit(10);

    for(int t = 0; t < epochs; t++) {
        for (int n = 0; n < 1000; n++) {
            length = generator.randomInt(1, 3);
            for(int i = 0; i < length; i++) {
                digit = generator.randomInt(0,9);
                NetworkUtils::binaryEncoding(digit, &binDigit);
                msom.train(binDigit.data());
                if (digit == 0 && i == 0) {
                    break;
                }
            }
            //msom.resetContext();
        }
        cout << "Epoch " << t << " qError: " << msom.getError() << " WD: " << msom.getWinnerDifferentiation() << endl;
        msom.paramDecay();
    }
}