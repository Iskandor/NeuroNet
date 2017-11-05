//
// Created by user on 17. 10. 2017.
//

#ifndef NEURONET_EGREEDY_H
#define NEURONET_EGREEDY_H


#include "IExploration.h"

namespace NeuroNet {

class eGreedy : public IExploration {

public:
    eGreedy(double p_startE, double p_endE);
    virtual ~eGreedy() {};

    int chooseAction(Vector* p_values);

    void update(double f);

private:
    double _startE, _endE;
    double _epsilon;

};

}

#endif //NEURONET_EGREEDY_H
