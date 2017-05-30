//
// Created by user on 24. 5. 2017.
//

#ifndef NEURONET_TICTACTOE_H
#define NEURONET_TICTACTOE_H

#include <vector>
#include <string>
#include <Environment.h>

using namespace std;

namespace NeuroNet {

class TicTacToe : public Environment {
public:
    TicTacToe(int p_dim);
    ~TicTacToe();

    vector<double> getSensors();
    void performAction(double p_action);
    void reset();

    inline int getField(int i, int j) { return _board[i * _dim + j]; };
    inline bool invalidMove() { return _invalidMove; };
    inline int activePlayer() {return _activePlayer; };

    string toString();

private:
    void changeTurn();

private:
    int _dim;
    int *_board;

    int _activePlayer;

    bool _invalidMove;
};

}


#endif //NEURONET_TICTACTOE_H
