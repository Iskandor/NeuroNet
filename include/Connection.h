#pragma once

#include "NeuralGroup.cuh"
#include <vector>
#include <matrix2.h>

using namespace std;

class Connection
{
public:
    Connection(int p_id, NeuralGroup* p_inGroup, NeuralGroup* p_outGroup);
    ~Connection(void);

public:
    void init(double p_density, double p_inhibition) const;
    matrix2<double>* getWeights() const { return _weights; };

    NeuralGroup* getOutGroup() const { return _outGroup; };
    NeuralGroup* getInGroup() const { return _inGroup; };
    int getId() const { return _id; };

private:
    int _id;
    NeuralGroup* _inGroup;
    NeuralGroup* _outGroup;
    int _inDim, _outDim;
    matrix2<double>* _weights;
};