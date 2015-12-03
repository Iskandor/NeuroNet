#pragma once

#include "NeuralGroup.h"
#include <vector>
#include <matrix.h>

using namespace std;

class Connection
{
public:
    Connection(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, int p_speed);
    ~Connection(void);

public:
    void init(double p_density, double p_inhibition);
    matrix<double>* getWeights() { return _weights; };

    NeuralGroup* getOutGroup() { return _outGroup; };
    NeuralGroup* getInGroup() { return _inGroup; };

    void addSignal(double* p_signal);
    void loopSignal();
    double* getSignal();

private:
    NeuralGroup* _inGroup;
    NeuralGroup* _outGroup;
    int _inDim, _outDim;
    int _speed;
    matrix<double>* _weights;
    vector<pair<int, double*>> _signals;
};