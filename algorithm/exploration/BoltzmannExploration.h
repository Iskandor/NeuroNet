//
// Created by user on 17. 10. 2017.
//

#ifndef NEURONET_BOLTZMANNEXPLORATION_H
#define NEURONET_BOLTZMANNEXPLORATION_H


#include "IExploration.h"

namespace NeuroNet {

class BoltzmannExploration : public IExploration {

public:
    BoltzmannExploration(double p_startT, double p_endT);
    virtual ~BoltzmannExploration() {};

    int chooseAction(Vector* p_values);

    void update(double f);

private:
    double _startT, _endT;
    double _temperature;

};
}



#endif //NEURONET_BOLTZMANNEXPLORATION_H
