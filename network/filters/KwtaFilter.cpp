//
// Created by user on 25. 3. 2016.
//

#include "KwtaFilter.h"

KwtaFilter::KwtaFilter(int p_k, bool p_binaryVector) : IFilter() {
  _k = p_k;
  _binaryVector = p_binaryVector;
}

KwtaFilter::~KwtaFilter() {

}

VectorXd &KwtaFilter::process(VectorXd *p_input) {
  VectorXd temp(p_input->size());
  int k = _k;
  double max;
  _output = VectorXd::Zero(p_input->size());
  temp << *p_input;

  while(k > 0) {
    max = temp.maxCoeff();
    for(int i = 0; i < p_input->size(); i++) {
      if (temp[i] == max) {
        _output[i] = _binaryVector ? 1 : temp[i];
        temp[i] = -INFINITY;
        k--;
      }
    }
  }

  return _output;
}
