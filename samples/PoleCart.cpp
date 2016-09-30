//
// Created by mpechac on 9. 6. 2016.
//

#include "PoleCart.h"
#include "../network/NetworkUtils.h"
#include "../network/Define.h"

PoleCart::PoleCart() {
    reset();
    _force = 0.1;
}

PoleCart::~PoleCart() {

}

bool PoleCart::evaluateAction(VectorXd *p_action, VectorXd *p_state) {
    return false;
}

void PoleCart::updateState(VectorXd *p_action) {
    _force = FORCE * (*p_action)[0];
    update();
    _time++;

    _reward = 1;
    /*
    if (isFinished()) {
        _reward = 2 * (_time * .001) - 1;
    }
    */
    if (isFailed()) {
        _reward = -1;
        //_reward = 2 * (_time * .001) - 1;
    }
}

VectorXd *PoleCart::getState() {
    VectorXd position = VectorXd::Zero(20);
    VectorXd angle = VectorXd::Zero(20);
    VectorXd positionV = VectorXd::Zero(20);
    VectorXd angleV = VectorXd::Zero(20);

    _neuralState = VectorXd::Zero(80);

    NetworkUtils::gaussianEncoding(_x / track_limit, -1, 1, 20, &position);
    NetworkUtils::gaussianEncoding(_th * 15 / PI, -1, 1, 20, &angle);
    NetworkUtils::gaussianEncoding(_xVel / track_limit, -1, 1, 20, &positionV);
    NetworkUtils::gaussianEncoding(_thVel * 15 / PI, -1, 1, 20, &angleV);

    _neuralState << position, angle, positionV, angleV;

    return &_neuralState;
}

void PoleCart::reset() {
    _time = 0;
    _x = 0;
    _th = 0;
    _xVel = _xAcc = _thVel = _thAcc = 0;
    _force = 0;
}

int PoleCart::getStateSize() {
    return 80;
}

bool PoleCart::isFinished() const {
    return _time == 195;
}

bool PoleCart::isFailed() const {
    return (abs(_x) > track_limit || abs(_th) > pole_failure);
}

void PoleCart::update() {
    _thAcc = (gravity * sin(_th) + cos(_th) * ( (-_force - mass_cart * pole_length * pow(_thVel, 2) * sin(_th)) / (mass_cart + mass_pole) )) / (pole_length * (4/3 - (mass_pole * pow(cos(_th), 2))/(mass_cart + mass_pole)));
    _xAcc = (_force + mass_pole*pole_length*(pow(_thVel,2)*sin(_th) - _thAcc*cos(_th))) / (mass_cart + mass_pole);
    _thVel += dt * _thAcc;
    _xVel += dt * _xAcc;
    _th += dt * _thVel;
    _x += dt * _xVel;
}

void PoleCart::print(ostream &os) {
    os << "Position :" << _x << endl;
    os << "Angle   :" << _th << endl;
    os << "Force   :" << _force << endl;
}
