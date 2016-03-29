//
// Created by user on 25. 3. 2016.
//

#ifndef NEURONET_IFILTER_H
#define NEURONET_IFILTER_H

#include <Eigen/Dense>

using namespace Eigen;

class IFilter {
public:
    IFilter() {};
    ~IFilter() {};

    virtual VectorXd& process(VectorXd* p_input) = 0;
protected:
    VectorXd _output;
};


#endif //NEURONET_IFILTER_H
