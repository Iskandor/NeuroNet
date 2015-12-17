#pragma once

#include "NeuralGroup.h"
#include <vector>
#include <matrix2.h>

using namespace std;

class Connection
{
public:
    Connection(int p_id, NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, int p_speed);
    ~Connection(void);

public:
    void init(double p_density, double p_inhibition);
    matrix2<double>* getWeights() { return _weights; };

    NeuralGroup* getOutGroup() { return _outGroup; };
    NeuralGroup* getInGroup() { return _inGroup; };
    int getId() { return _id; };

    void addSignal(double* p_signal);
    void loopSignal();
    double* getSignal();

private:
    int _id;
    NeuralGroup* _inGroup;
    NeuralGroup* _outGroup;
    int _inDim, _outDim;
    int _speed;
    matrix2<double>* _weights;
    vector<pair<int, double*>> _signals;
};