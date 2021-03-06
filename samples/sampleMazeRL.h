//
// Created by mpechac on 22. 3. 2017.
//

#ifndef NEURONET_SAMPLEMAZERL_H
#define NEURONET_SAMPLEMAZERL_H

#include <vector>
#include <Vector.h>
#include <algorithm/exploration/IExploration.h>

using namespace FLAB;
using namespace std;

class sampleMazeRL {
public:
    sampleMazeRL();
    ~sampleMazeRL();
    void sampleQ();
    void sampleSARSA();
    void sampleAC();
    void sampleTD();
private:
    Vector encodeState(vector<double> *p_sensors);
    int chooseAction(Vector* p_input, double epsilon);

    IExploration* exploration;
};

#endif //NEURONET_SAMPLEMAZERL_H
