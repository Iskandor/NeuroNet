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

    double normalRandom(double p_sigma);
    int random(int p_lower, int p_upper);
    double random(double p_lower = 0, double p_upper = 1);

private:
    RandomGenerator();
    std::random_device _rd;
    std::mt19937 _mt;
};

}

#endif //NEURONET_RANDOMGENERATOR_H
