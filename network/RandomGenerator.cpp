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
  return (*_dist)(_mt);
  //return static_cast<double>(rand()) / RAND_MAX;
}

int RandomGenerator::random(int p_lower, int p_upper) {
  return p_lower + (int)round(random() * (p_upper - p_lower));
}

double RandomGenerator::random(double p_lower, double p_upper) {
  return p_lower + random() * (p_upper - p_lower);
}

RandomGenerator& RandomGenerator::getInstance() {
  static RandomGenerator instance;
  return instance;
}
