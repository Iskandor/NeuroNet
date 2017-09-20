#pragma once
#include <vector>
#include "NeuralGroup.h"

using namespace std;
using json = nlohmann::json;

namespace NeuroNet {

class Connection
{
public:
    enum INIT {
        UNIFORM = 0,
        LECUN_UNIFORM = 1,
        GLOROT_UNIFORM = 2,
        IDENTITY = 3
    };

    Connection(int p_id, NeuralGroup* p_inGroup, NeuralGroup* p_outGroup);
    Connection(Connection& p_copy);
    ~Connection(void);

    void init(INIT p_init, double p_limit);
    void init(double p_density, double p_inhibition) const;
    void init(Matrix* p_weights);
    void setWeights(Matrix* p_weights);
    Matrix* getWeights() const { return _weights; };

    NeuralGroup* getOutGroup() const { return _outGroup; };
    NeuralGroup* getInGroup() const { return _inGroup; };
    int getId() const { return _id; };

    json getFileData();
private:
    void uniform(double p_limit);
    void identity();


    int _id;
    NeuralGroup* _inGroup;
    NeuralGroup* _outGroup;
    int _inDim, _outDim;
    Matrix* _weights;
};

}