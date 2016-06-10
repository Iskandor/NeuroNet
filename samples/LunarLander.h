//
// Created by user on 28. 2. 2016.
//

#ifndef LIBNEURONET_LUNARLANDER_H
#define LIBNEURONET_LUNARLANDER_H

#include <iostream>
#include "../algorithm/IEnvironment.h"

using namespace std;
using namespace NeuroNet;

class LunarLander : public IEnvironment {
public:
    const double dt = 1;
    const double gravity = 0.5;
    const double engine_strength = 1.0;
    const double safe_velocity = -0.5;

    LunarLander();

    void print(ostream &os);

    bool evaluateAction(VectorXd *p_action, VectorXd *p_state) override;

    void updateState(VectorXd *p_action) override;

    VectorXd *getState() override;

    void reset() override;

    bool isFinished() const;

    int getStateSize() override;

private:
    void decodeAction(VectorXd *p_action, double &p_command) const;

    void update(double rate);

    double _height;
    double _velocity;
    double _fuel;
    VectorXd _neuralState;
};

#endif //LIBNEURONET_LUNARLANDER_H
