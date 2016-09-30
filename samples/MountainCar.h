//
// Created by mpechac on 14. 6. 2016.
//

#ifndef NEURONET_MOUNTAINCAR_H
#define NEURONET_MOUNTAINCAR_H

#include "../algorithm/IEnvironment.h"

using namespace NeuroNet;

class MountainCar : public IEnvironment {
public:
    const double dt = 0.01;
    const double gravity = -9.81; // gravity of the environment
    const double max_time = 100;     // max time per episode (task is terminated after this)
    const double mass = 0.2;        // mass of the car
    const double fric = 0.3;        // friction factor
    const double lim_velocity = 0.07;
    const double llim_position = -1.2;
    const double rlim_position = 0.6;

    MountainCar();
    ~MountainCar();

    bool evaluateAction(VectorXd *p_action, VectorXd *p_state) override;
    void updateState(VectorXd *p_action) override;
    VectorXd *getState() override;
    void reset() override ;
    int getStateSize() override;

    bool isFinished() const;
    bool isFailed() const;

private:
    void decodeAction(VectorXd* p_action);
    void update();

    int _time;
    double _motor;
    double _position;
    double _velocity;

    VectorXd _neuralState;
};


#endif //NEURONET_MOUNTAINCAR_H
