//
// Created by user on 28. 2. 2016.
//

#include "LunarLander.h"
#include "../network/NetworkUtils.h"

//Class definitions
LunarLander::LunarLander()
{
    _state = VectorXd::Zero(3);
    reset();
}

void LunarLander::print(ostream& os)
{
    os << "Velocity :" << _velocity << endl;
    os << "Height   :" << _height << endl;
    os << "Fuel     :" << _fuel << endl;
}

void LunarLander::update(double rate)
{
    double dh;
    double dv;
    if (_fuel <= 0) {
        _fuel =0;
        rate=0;
    }
    dh= _velocity * dt;
    dv= engine_strength*rate-gravity;
    _fuel -= (rate * dt);
    _height += dh;
    _velocity +=dv;

    _state[0] = _fuel;
    _state[1] = _height;
    _state[2] = _velocity;
}

bool LunarLander::evaluateAction(VectorXd *p_action, VectorXd *p_state) {
    return true;
}

void LunarLander::updateState(VectorXd *p_action) {
    double rate = 0;
    decodeAction(p_action, rate);
    update(rate);
    _reward = 0;
    if (isFinished()) {
        cout<<"Final velocity: "<<_velocity;
        if (_velocity >= safe_velocity) {
            _reward = 10;
            cout<<"...good landing!\n";
        }
        else {
            _reward = (_velocity - safe_velocity) / 35;
            cout<<"...you crashed!\n";
        }
    }
}

void LunarLander::reset() {
    _height = 50.0;
    _velocity = 0.0;
    _fuel = 20.0;
}

bool LunarLander::isFinished() const {
    if (_height <= 0)
        return true;
    return false;
}

void LunarLander::decodeAction(VectorXd *p_action, double &p_command) const {
    if ((*p_action)[0] == 1) {
        p_command = 0;
    }
    if ((*p_action)[1] == 1) {
        p_command = 1;
    }
}

VectorXd *LunarLander::getState() {
    VectorXd height = VectorXd::Zero(50);
    VectorXd velocity = VectorXd::Zero(20);
    VectorXd fuel = VectorXd::Zero(20);
    _neuralState = VectorXd::Zero(50+20+20);

    NetworkUtils::gaussianEncoding(_height, 50, 0, 50, &height);
    NetworkUtils::gaussianEncoding(_velocity, 35, 0, 20, &velocity);
    NetworkUtils::gaussianEncoding(_fuel, 20, 0, 20, &fuel);

    _neuralState << height, velocity, fuel;

    return &_neuralState;
}

int LunarLander::getStateSize() {
    return 90;
}
