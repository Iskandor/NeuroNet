//
// Created by user on 14. 5. 2016.
//

#ifndef NEURONET_RANDOMGENERATOR_H
#define NEURONET_RANDOMGENERATOR_H


#include <random>

namespace NeuroNet {

class RandomGenerator {
public:
    RandomGenerator();
    ~RandomGenerator();

    double random();

private:
    std::random_device _rd;
    std::mt19937 _mt;
    std::uniform_real_distribution<double> *_dist;
};

}

#endif //NEURONET_RANDOMGENERATOR_H
