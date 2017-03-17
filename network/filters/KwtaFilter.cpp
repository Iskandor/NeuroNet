//
// Created by user on 25. 3. 2016.
//

#include <math.h>
#include "KwtaFilter.h"

using namespace NeuroNet;

KwtaFilter::KwtaFilter(int p_k, bool p_binaryVector) : IFilter() {
  _k = p_k;
  _binaryVector = p_binaryVector;
}

KwtaFilter::~KwtaFilter() {

}

Vector &KwtaFilter::process(Vector *p_input) {
  Vector temp = Vector(*p_input);
  int k = _k;
  double max;
  _output = Vector::Zero(p_input->size());

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
