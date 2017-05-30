//
// Created by user on 24. 5. 2017.
//

#include "TicTacToe.h"
#include <RandomGenerator.h>

using namespace NeuroNet;
using namespace FLAB;

TicTacToe::TicTacToe(int p_dim) {
    _dim = p_dim;
    _board = new int[_dim * _dim];
    reset();
}

vector<double> TicTacToe::getSensors() {
    vector<double> res;

    for(int i = 0; i < _dim; i++) {
        for(int j = 0; j < _dim; j++) {
            res.push_back(_board[i * _dim + j]);
        }
    }

    return vector<double>(res);
}

void TicTacToe::performAction(double p_action) {
    _invalidMove = false;

    if (_board[(int)p_action] == 0) {
        _board[(int)p_action] = _activePlayer;
    }
    else {
        _invalidMove = true;
    }

    changeTurn();
}

void TicTacToe::reset() {
    _activePlayer = RandomGenerator::getInstance().random(0, 1);
    changeTurn();

    for(int i = 0; i < _dim; i++) {
        for(int j = 0; j < _dim; j++) {
            _board[i * _dim + j] = 0;
        }
    }
}

void TicTacToe::changeTurn() {
    if (_activePlayer == 1) {
        _activePlayer = -1;
    }
    else {
        _activePlayer = 1;
    }
}

string TicTacToe::toString() {
    string s;

    for(int i = 0; i < _dim; i++) {
        for(int j = 0; j < _dim; j++) {
            if (_board[i * _dim + j] == 0) {
                s += ' ';
            }
            if (_board[i * _dim + j] == -1) {
                s += 'X';
            }
            if (_board[i * _dim + j] == 1) {
                s += 'O';
            }
        }
        s += '\n';
    }

    return s;
}

TicTacToe::~TicTacToe() {
    delete[] _board;
}
