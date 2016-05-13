//
// Created by user on 25. 3. 2016.
//

#include "NormalizationFilter.h"
#include "../NetworkUtils.h"

using namespace NeuroNet;

NormalizationFilter::NormalizationFilter(VectorXd* p_limit) : IFilter() {
  _limit.resize(p_limit->size());
  _limit << *p_limit;
}

NormalizationFilter::~NormalizationFilter() {

}

VectorXd &NormalizationFilter::process(VectorXd *p_input) {
  _output.resize(p_input->size());
  _output << *p_input;

  for(int i = 0; i < p_input->size(); i++) {
    _output[i] = fabs(_output[i]) / _limit[i] * NetworkUtils::sgn(_output[i]);
  }

  return _output;
}
