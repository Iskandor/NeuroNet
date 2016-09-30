//
// Created by mpechac on 9. 6. 2016.
//

#ifndef NEURONET_POLECART_H
#define NEURONET_POLECART_H

#include "../algorithm/IEnvironment.h"

using namespace NeuroNet;
using namespace std;

class PoleCart : public IEnvironment {
public:
    const double dt = 0.02;
    const double gravity = 9.81;
    const double mass_cart = 1.0;
    const double mass_pole = 0.1;
    const double pole_length = 0.5;
    const double pole_failure = 0.209;
    const double track_limit = 2.4;
    const double FORCE = 10;

    PoleCart();
    ~PoleCart();

    bool evaluateAction(VectorXd *p_action, VectorXd *p_state) override;
    void updateState(VectorXd *p_action) override;
    VectorXd *getState() override;
    void reset() override ;
    int getStateSize() override;

    bool isFinished() const;
    bool isFailed() const;

    void print(ostream &os);

    double getTheta() { return _th; };

private:
    void update();
    //void decodeAction(VectorXd *p_action, double &p_command) const;

    int _time;
    double _x;
    double _xVel;
    double _xAcc;
    double _th;
    double _thVel;
    double _thAcc;
    double _force;

    VectorXd _neuralState;
};


#endif //NEURONET_POLECART_H
