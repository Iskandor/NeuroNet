//
// Created by user on 17. 10. 2017.
//

#ifndef NEURONET_IEXPLORATION_H
#define NEURONET_IEXPLORATION_H

#include <Vector.h>

using namespace FLAB;

namespace NeuroNet {

class IExploration {
public:
    IExploration() {};
    virtual ~IExploration() {};

    virtual int chooseAction(Vector* p_values) = 0;
    virtual void update(double f) {};

protected:
    inline double linterp(double a, double b, double f)
    {
        return (a * (1.0 - f)) + (b * f);
    };
};

}



#endif //NEURONET_IEXPLORATION_H
