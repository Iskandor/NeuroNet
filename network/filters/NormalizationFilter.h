//
// Created by user on 25. 3. 2016.
//

#ifndef NEURONET_NORMALIZATIONFILTER_H
#define NEURONET_NORMALIZATIONFILTER_H

#include "IFilter.h"

namespace NeuroNet {

class NormalizationFilter : public IFilter {
public:
  NormalizationFilter(VectorXd* p_limit);
  ~NormalizationFilter();

  VectorXd& process(VectorXd* p_input);
private:
  VectorXd _limit;
};

}
#endif //NEURONET_NORMALIZATIONFILTER_H
