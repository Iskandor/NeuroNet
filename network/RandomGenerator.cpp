//
// Created by user on 14. 5. 2016.
//

#include "RandomGenerator.h"

using namespace NeuroNet;

RandomGenerator::RandomGenerator() {
  _mt.seed(_rd());
  _dist = new std::uniform_real_distribution<double>(0,1);
}

RandomGenerator::~RandomGenerator() {
  delete _dist;
}

double RandomGenerator::random() {
  //return (*_dist)(_mt);
  return static_cast<double>(rand()) / RAND_MAX;
}
