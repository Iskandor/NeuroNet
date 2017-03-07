//
// Created by mpechac on 7. 3. 2017.
//

#include <c++/iostream>
#include "MazeTask.h"
#include "../network/RandomGenerator.h"

MazeTask::MazeTask() {
    int topology[] = {1, 1, 1, 1, 1,
                      1, 0, 0, 0, 1,
                      1, 1, 0, 1, 1,
                      1, 0, 0, 0, 1,
                      1, 1, 1, 1, 1};

    maze = new Maze(topology, 5, 5, 18);
}

void MazeTask::run() {

    int action;

    for(int e = 0; e < 100; e++) {
        cout << e << endl;
        action = RandomGenerator::getInstance().random(0, 3);
        cout << maze->toString() << endl;
        maze->performAction(action);
        cout << getReward() << endl;
    }
    cout << maze->toString() << endl;
}

MazeTask::~MazeTask() {
    delete maze;
}

bool MazeTask::isFinished() {
    return maze->actor() == maze->goal();
}

double MazeTask::getReward() {
    double reward = defautPenalty;

    if (isFinished()) reward = finalReward;
    if (maze->bang()) reward = bangPenalty;

    return reward;
}
