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

    inline Maze *getEnvironment() {
        return maze;
    }


private:
    const double defautPenalty = -1;
    const double bangPenalty = -1;
    const double killPenalty = -10;
    const double finalReward = 10;

    Maze *maze;
};


#endif //NEURONET_MAZETASK_H
