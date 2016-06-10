//
// Created by user on 14. 5. 2016.
//

#ifndef NEURONET_RANDOMGENERATOR_H
#define NEURONET_RANDOMGENERATOR_H


#include <random>

namespace NeuroNet {

class RandomGenerator {
public:
    static RandomGenerator& getInstance();
    RandomGenerator(RandomGenerator const&) = delete;
    void operator=(RandomGenerator const&)  = delete;
    ~RandomGenerator();

    double random();
    int random(int p_lower, int p_upper);
    double random(double p_lower, double p_upper);

private:
    RandomGenerator();
    std::random_device _rd;
    std::mt19937 _mt;
    std::uniform_real_distribution<double> *_dist;
};

}

#endif //NEURONET_RANDOMGENERATOR_H
