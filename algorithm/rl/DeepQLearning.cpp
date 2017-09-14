//
// Created by user on 30. 8. 2017.
//

#include <RandomGenerator.h>
#include "DeepQLearning.h"

using namespace NeuroNet;

DeepQLearning::DeepQLearning(NeuroNet::Optimizer *p_optimizer, NeuralNetwork* p_network, double p_gamma) : QLearning(p_optimizer, p_network, p_gamma) {
}

DeepQLearning::~DeepQLearning() {
}

double DeepQLearning::train(Vector *p_state0, int p_action0, Vector *p_state1, double p_reward) {

    if (_memory.size() > _memorySize) {
        _memory.erase(_memory.begin());
    }

    if (_batch >= _batchSize && _memory.size() >= _batchSize ) {
        _batch = 0;
        fillTrainingList();

        for(auto it = _trainingList.begin(); it != _trainingList.end(); it++) {
            BufferElem data = *it;
            QLearning::train(&data.s0, data.action, &data.s1, data.reward);
        }
    }
    else {
        _memory.push_back(BufferElem(*p_state0, p_action0, *p_state1, p_reward));
        _batch++;
    }

    return 0;
}

void DeepQLearning::init(double p_alpha, int p_batchSize, int p_memorySize) {
    _optimizer->init(p_alpha);
    _batchSize = p_batchSize;
    _memorySize = p_memorySize;
    _batch = 0;
}

DeepQLearning::BufferElem::BufferElem(Vector p_s0, int p_a, Vector p_s1, double p_r) {
    s0 = p_s0;
    action = p_a;
    s1 = p_s1;
    reward = p_r;
}

void DeepQLearning::fillTrainingList() {
    _trainingList.clear();

    for(int i = 0; i < _batchSize; i++) {
        int index = (unsigned int) RandomGenerator::getInstance().random(0, _batchSize - 1);
        _trainingList.push_back(_memory[index]);
    }
}
