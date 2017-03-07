//
// Created by mpechac on 7. 3. 2017.
//

#ifndef NEURONET_MAZETASK_H
#define NEURONET_MAZETASK_H


#include "../environments/maze/Maze.h"

using namespace NeuroNet;

class MazeTask {
public:
    MazeTask();
    ~MazeTask();

    void run();

    bool isFinished();
    double getReward();


private:
    const double defautPenalty = 0;
    const double bangPenalty = 0;
    const double finalReward = 1;

    Maze *maze;
};


#endif //NEURONET_MAZETASK_H
