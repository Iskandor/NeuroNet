//
// Created by user on 25. 3. 2016.
//

#include "NormalizationFilter.h"
#include "../NetworkUtils.h"

using namespace NeuroNet;

NormalizationFilter::NormalizationFilter(Vector* p_limit) : IFilter() {
  _limit = Vector(*p_limit);
}

NormalizationFilter::~NormalizationFilter() {

}

Vector &NormalizationFilter::process(Vector *p_input) {
  _output = Vector(*p_input);

  for(int i = 0; i < p_input->size(); i++) {
    _output[i] = fabs(_output[i]) / _limit[i] * NetworkUtils::sgn(_output[i]);
  }

  return _output;
}
