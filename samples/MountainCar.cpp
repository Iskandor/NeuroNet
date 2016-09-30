//
// Created by mpechac on 14. 6. 2016.
//

#include "MountainCar.h"
#include "../network/NetworkUtils.h"

MountainCar::MountainCar() {
    reset();
}

MountainCar::~MountainCar() {

}

bool MountainCar::evaluateAction(VectorXd *p_action, VectorXd *p_state) {
    return false;
}

void MountainCar::updateState(VectorXd *p_action) {
    decodeAction(p_action);
    update();

    _reward = 0;
    if (isFinished()) {
        _reward = 1;
    }
    if (isFailed()) {
        _reward = -1;
    }
}

VectorXd *MountainCar::getState() {
    VectorXd position = VectorXd::Zero(100);
    VectorXd velocity = VectorXd::Zero(100);
    _neuralState = VectorXd::Zero(200);

    NetworkUtils::gaussianEncoding(_position, llim_position, rlim_position, 100, &position);
    NetworkUtils::gaussianEncoding(_velocity, -lim_velocity, lim_velocity, 100, &velocity);

    _neuralState << position, velocity;

    return &_neuralState;
}

void MountainCar::reset() {
    _time = 0;
    _position = 0.5;
    _velocity = 0;
}

int MountainCar::getStateSize() {
    return 200;
}

bool MountainCar::isFinished() const {
    return _position > rlim_position;
}

bool MountainCar::isFailed() const {
    return _time > max_time || _position < llim_position;
}

void MountainCar::update() {
    _time++;
    _velocity = _velocity + _motor * 0.001 + cos(3*_position) * -0.0025;
    _position = _position + _velocity;
}

void MountainCar::decodeAction(VectorXd *p_action) {
    if ((*p_action)(0) == 1) { // left
        _motor = -1;
    }
    if ((*p_action)(1) == 1) { // neutral
        _motor = 0;
    }
    if ((*p_action)(2) == 1) { // right
        _motor = 1;
    }
}
