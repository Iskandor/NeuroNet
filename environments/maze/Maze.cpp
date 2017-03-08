//
// Created by mpechac on 7. 3. 2017.
//

#include <c++/iostream>
#include "Maze.h"
#include "../../network/RandomGenerator.h"


using namespace NeuroNet;

Maze::Maze(int *p_topology, int p_mazeX, int p_mazeY, int p_goal) {
    for (int i = 0; i < p_mazeX * p_mazeY; i++) {
        _mazeTable.push_back(p_topology[i]);
    }

    _goal = p_goal;

    _mazeX = p_mazeX;
    _mazeY = p_mazeY;

    vector<int> free = freePos();

    for (int i = 0; i < free.size(); i++) {
        if (free[i] != _goal) _initPos.push_back(free[i]);
    }

    _numActions = 4;
    _actions.push_back(MazeAction("N", 0, -1));
    _actions.push_back(MazeAction("E", 1, 0));
    _actions.push_back(MazeAction("S", 0, 1));
    _actions.push_back(MazeAction("W", -1, 0));

    reset();
}

Maze::~Maze() {

}

void Maze::reset() {
    _bang = false;
    _actor = RandomGenerator::getInstance().choice(&_initPos)[0];
}

vector<int> Maze::freePos() {
    vector<int> res;

    for(unsigned int i = 0; i < _mazeTable.size(); i++) {
        if (_mazeTable[i] == 0) {
            res.push_back(i);
        }
    }

    return vector<int>(res);
}

int Maze::moveInDir(int p_x, int p_y) {
    return _actor + p_y * _mazeX + p_x;
}

void Maze::performAction(int p_action) {
    //cout << _actions[p_action].Id() << endl;
    int newPos = moveInDir(_actions[p_action].X(), _actions[p_action].Y());

    if (newPos < 0 || newPos > _mazeTable.size()) {
        _bang = true;
    }
    else {
        if (_mazeTable[newPos] == 0) {
            _actor = newPos;
            _bang = false;
        }
        else {
            _bang = true;
        }
    }
}

vector<int> Maze::getSensors() {
    vector<int> res(_numActions, 0);
    int obs;

    for(int i = 0; i < _numActions; i++) {
        obs = moveInDir(_actions[i].X(), _actions[i].Y());
        if (obs > 0 && obs < _mazeTable.size()) {
            res[i] = _mazeTable[(unsigned int) obs];
        }
    }

    return vector<int>(res);
}


string Maze::toString() {
    string s;

    int index;

    for(int i = 0; i < _mazeY; i++) {
        for(int j = 0; j < _mazeX; j++) {
            index = i * _mazeY + j;
            if (_actor == index) {
                s += "@";
            }
            else if (_goal == index) {
                s += "*";
            }
            else if (_mazeTable[index] == 0) {
                s += "O";
            }
            else {
                s += "#";
            }
        }
        s += '\n';
    }

    return s;
}
