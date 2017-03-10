//
// Created by mpechac on 9. 3. 2017.
//

#ifndef NEURONET_RMSPROP_H
#define NEURONET_RMSPROP_H


#include "GradientDescent.h"

namespace NeuroNet {

class RMSProp : public GradientDescent {

public:
    RMSProp(NeuralNetwork *p_network, double p_cacheDecay, double p_momentum, bool p_nesterov);
    ~RMSProp();

    double train(VectorXd *p_input, VectorXd* p_target);

protected:
    void    updateBatch();
    double  calcMse(VectorXd *p_target);
    void update(NeuralGroup* p_node);
    void updateWeights(Connection* p_connection);

private:
    void weightDecay(Connection* p_connection) const;

protected:
    double  _weightDecay;
    double  _cacheDecay;
    VectorXd _error;
    map<int, MatrixXd> _eps;
    map<int, MatrixXd> _gradientCache;
};

}

#endif //NEURONET_RMSPROP_H
