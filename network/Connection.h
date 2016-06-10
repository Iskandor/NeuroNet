#pragma once
#include <vector>
#include "NeuralGroup.h"

using namespace std;
using json = nlohmann::json;

namespace NeuroNet {

class Connection
{
public:
    Connection(int p_id, NeuralGroup* p_inGroup, NeuralGroup* p_outGroup);
    ~Connection(void);

public:
    void init(double p_limit);
    void init(double p_density, double p_inhibition) const;
    void init(MatrixXd* p_weights);
    MatrixXd* getWeights() const { return _weights; };

    NeuralGroup* getOutGroup() const { return _outGroup; };
    NeuralGroup* getInGroup() const { return _inGroup; };
    int getId() const { return _id; };

    json getFileData();
private:
    int _id;
    NeuralGroup* _inGroup;
    NeuralGroup* _outGroup;
    int _inDim, _outDim;
    MatrixXd* _weights;
};

}