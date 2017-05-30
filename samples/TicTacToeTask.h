//
// Created by user on 24. 5. 2017.
//

#ifndef NEURONET_TICTACTOETASK_H
#define NEURONET_TICTACTOETASK_H


#include <TicTacToe.h>

namespace NeuroNet {

class TicTacToeTask {
public:
    TicTacToeTask();
    ~TicTacToeTask();

    void run();

    bool isFinished();
    double getReward(int p_player);

    inline TicTacToe *getEnvironment() {
        return _game;
    }

    inline int Winner() { return _winner; };


private:
    int _winner;

    const double defautReward = 0;
    const double invalidPenalty = -1;
    const double looseReward = -10;
    const double winReward = 10;

    TicTacToe *_game;
};

}

#endif //NEURONET_TICTACTOETASK_H
