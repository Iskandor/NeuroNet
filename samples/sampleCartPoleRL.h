//
// Created by mpechac on 22. 3. 2017.
//

#ifndef NEURONET_SAMPLECARTPOLERL_H
#define NEURONET_SAMPLECARTPOLERL_H

#include <vector>
#include "../backend/FLAB/Vector.h"

using namespace FLAB;
using namespace std;

class sampleCartPoleRL {
public:
    sampleCartPoleRL();
    ~sampleCartPoleRL();

    void sampleCACLA();
private:
    Vector encodeState(vector<double> *p_sensors);
    Vector chooseAction(Vector &p_action, double epsilon);
};

#endif //NEURONET_SAMPLECARTPOLERL_H
