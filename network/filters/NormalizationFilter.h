//
// Created by user on 25. 3. 2016.
//

#ifndef NEURONET_NORMALIZATIONFILTER_H
#define NEURONET_NORMALIZATIONFILTER_H

#include "IFilter.h"

namespace NeuroNet {

class NormalizationFilter : public IFilter {
public:
  NormalizationFilter(Vector* p_limit);
  ~NormalizationFilter();

  Vector& process(Vector* p_input);
private:
  Vector _limit;
};

}
#endif //NEURONET_NORMALIZATIONFILTER_H
