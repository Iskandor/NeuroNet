//
// Created by user on 25. 3. 2016.
//

#ifndef NEURONET_KWTAFILTER_H
#define NEURONET_KWTAFILTER_H


#include "IFilter.h"

class KwtaFilter : public IFilter {
public:
    KwtaFilter(int p_k);
    ~KwtaFilter();

    VectorXd& process(VectorXd* p_input);

private:
    int _k;
};


#endif //NEURONET_KWTAFILTER_H
