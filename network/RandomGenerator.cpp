//
// Created by user on 14. 5. 2016.
//

#include "RandomGenerator.h"

using namespace NeuroNet;

RandomGenerator::RandomGenerator() {
  _mt.seed(_rd());
}

RandomGenerator::~RandomGenerator() {
}

RandomGenerator& RandomGenerator::getInstance() {
  static RandomGenerator instance;
  return instance;
}

int RandomGenerator::random(int p_lower, int p_upper) {
  std::uniform_int_distribution<int> distribution(p_lower, p_upper);
  return distribution(_mt);
}

double RandomGenerator::random(double p_lower, double p_upper) {
  std::uniform_real_distribution<double> distribution(p_lower, p_upper);
  return distribution(_mt);
}

double RandomGenerator::normalRandom(double p_sigma) {
  std::normal_distribution<double> distribution(0, p_sigma);
  return distribution(_mt);
}
