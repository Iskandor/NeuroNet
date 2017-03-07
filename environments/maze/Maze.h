//
// Created by mpechac on 7. 3. 2017.
//

#ifndef NEURONET_MAZE_H
#define NEURONET_MAZE_H

#include <vector>
#include <map>
#include "../Environment.h"
#include "MazeAction.h"

using namespace std;

namespace NeuroNet {

class Maze : public Environment {
public:
    Maze(int* p_topology, int p_mazeX, int p_mazeY, int p_goal);
    ~Maze();

    vector<int> getSensors();
    void performAction(int p_action);
    void reset();

    string toString();

    inline int actor() {
        return _actor;
    }

    inline int goal() {
        return _goal;
    }

    inline bool bang() {
        return _bang;
    }


private:
    vector<int> freePos();
    int moveInDir(int p_x, int p_y);

private:
    int _mazeX, _mazeY;
    vector<int> _initPos;
    vector<int> _mazeTable;
    vector<MazeAction> _actions;

    int _actor;
    int _goal;
    bool _bang;

};

}

#endif //NEURONET_MAZE_H
