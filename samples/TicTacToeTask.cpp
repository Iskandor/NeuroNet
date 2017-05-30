//
// Created by user on 24. 5. 2017.
//

#include "TicTacToeTask.h"
#include <iostream>
#include <RandomGenerator.h>

using namespace NeuroNet;
using namespace FLAB;

TicTacToeTask::TicTacToeTask() {
    _game = new TicTacToe(3);
}

TicTacToeTask::~TicTacToeTask() {
    delete _game;
}

void TicTacToeTask::run() {
    int action;

    for(int e = 0; e < 100; e++) {
        cout << e << endl;
        action = RandomGenerator::getInstance().random(0, 8);
        cout << _game->toString() << endl;
        _game->performAction(action);
        cout << getReward(1) << endl;
        cout << getReward(-1) << endl;
    }
    cout << _game->toString() << endl;
}

bool TicTacToeTask::isFinished() {

    bool finished = false;
    int sum = 0;

    for(int i = 0; i < 3; i++) {
        sum = 0;
        for(int j = 0; j < 3; j++) {
            sum += _game->getField(i, j);
        }
        if (abs(sum) == 3) {
            finished = true;
            _winner = sum / 3;
        }
    }

    if (!finished) {
        for(int j = 0; j < 3; j++) {
            sum = 0;
            for(int i = 0; i < 3; i++) {
                sum += _game->getField(i, j);
            }
            if (abs(sum) == 3) {
                finished = true;
                _winner = sum / 3;
            }
        }
    }

    if (!finished) {
        sum = 0;
        for(int i = 0; i < 3; i++) {
            sum += _game->getField(i, i);
        }
        if (abs(sum) == 3) {
            finished = true;
            _winner = sum / 3;
        }
    }

    if (!finished) {
        sum = 0;
        for(int i = 0; i < 3; i++) {
            sum += _game->getField(2 - i, 2 - i);
        }
        if (abs(sum) == 3) {
            finished = true;
            _winner = sum / 3;
        }
    }

    if (!finished) {
        finished = true;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (_game->getField(i, j) == 0) {
                    finished = false;
                }
            }
        }
        if (finished) {
            _winner = 0;
        }
    }

    return finished;
}

double TicTacToeTask::getReward(int p_player) {
    double reward = defautReward;

    if (_game->invalidMove()) {
        reward = invalidPenalty;
    }
    if (isFinished() && _winner == p_player) {
        reward = winReward;
    }
    if (isFinished() && _winner != p_player) {
        reward = looseReward;
    }

    return reward;
}
