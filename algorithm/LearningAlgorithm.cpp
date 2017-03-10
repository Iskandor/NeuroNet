//
// Created by user on 1. 5. 2016.
//

#include "LearningAlgorithm.h"

using namespace NeuroNet;

LearningAlgorithm::LearningAlgorithm() {
  _alpha = 0;
  _batchSize = 1;
  _batch = 0;
}

LearningAlgorithm::~LearningAlgorithm() {

}

void LearningAlgorithm::setAlpha(double p_alpha) {
  _alpha = p_alpha;
}

void LearningAlgorithm::setBatchSize(int p_batchSize) {
  _batchSize = p_batchSize;
}

void LearningAlgorithm::updateBatch() {
    if (_batch < _batchSize) {
        _batch++;
    }
    else {
        _batch = 0;
    }
}
