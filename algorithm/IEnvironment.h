#pragma once

#include <Eigen/Dense>

using namespace Eigen;

namespace NeuroNet {

    class IEnvironment {
    public:
        IEnvironment() : _reward(0) { };

        virtual ~IEnvironment(void) { };

        virtual bool evaluateAction(VectorXd *p_action, VectorXd *p_state) = 0;

        virtual void updateState(VectorXd *p_action) = 0;

        virtual VectorXd *getState() { return &_state; };

        virtual void reset() = 0;

        double getReward() const { return _reward; };

        virtual int getStateSize() { return _state.size(); };

    protected:
        double _reward;
        VectorXd _state;
    };

}
