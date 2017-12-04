//
// Created by user on 5. 11. 2017.
//

#ifndef NEURONET_MNS_H
#define NEURONET_MNS_H

#include <MSOM.h>
#include "Dataset.h"

using namespace NeuroNet;

namespace MNS {

class ModelMNS {
public:
    ModelMNS();
    ~ModelMNS();

    void init();
    void run(int p_epochs);

    void test();
private:
    const int _sizePMC = 12;
    const int _sizeSTSp = 16;
    const int GRASPS = 3;
    const int PERSPS = 4;

    Dataset _data;
    MSOM    *_msomMotor;
    MSOM    *_msomVisual;
};

}

#endif //NEURONET_MNS_H
