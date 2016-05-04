//
// Created by user on 1. 5. 2016.
//

#include "LearningAlgorithm.h"

LearningAlgorithm::LearningAlgorithm() {
  _alpha = 0;
}

LearningAlgorithm::~LearningAlgorithm() {

}

void LearningAlgorithm::setAlpha(double p_alpha) {
  _alpha = p_alpha;
}
