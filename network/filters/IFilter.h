//
// Created by user on 25. 3. 2016.
//

#ifndef NEURONET_IFILTER_H
#define NEURONET_IFILTER_H

#include "../../backend/sflab/Vector.h"

using namespace SFLAB;

namespace NeuroNet {

class IFilter {
public:
    IFilter() {};
    ~IFilter() {};

    virtual Vector& process(Vector* p_input) = 0;
protected:
    Vector _output;
};

}

#endif //NEURONET_IFILTER_H
