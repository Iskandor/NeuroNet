//
// Created by user on 28. 5. 2017.
//

#ifndef NEURONET_SAMPLEGAMERL_H
#define NEURONET_SAMPLEGAMERL_H

#include <vector>
#include <Vector.h>
#include <QLearning.h>

using namespace FLAB;
using namespace std;
using namespace NeuroNet;

struct Player {
    int playerID;
    QLearning *agent;
};

class sampleGameRL {
public:
    sampleGameRL();
    ~sampleGameRL();

    void sampleTicTacToe();

private:
    Vector encodeState(int p_playerID, vector<double> *p_sensors);
    int chooseAction(Vector* p_input, double epsilon);
};

#endif //NEURONET_SAMPLEGAMERL_H
