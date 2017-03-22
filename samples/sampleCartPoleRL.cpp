//
// Created by mpechac on 21. 3. 2017.
//


#include "CartPoleTask.h"

void sampleContinuousRL() {
    CartPoleTask task;
    CartPole* cartPole = task.getEnvironment();

    task.run();
}