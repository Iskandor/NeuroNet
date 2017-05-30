//
// Created by user on 22. 5. 2017.
//

#ifndef NEURONET_AREADEFENSEGAME_H
#define NEURONET_AREADEFENSEGAME_H

#include <Environment.h>

namespace NeuroNet {

    class AreaDefenseGame : public Environment {

    public:
        AreaDefenseGame();

        virtual vector<double> getSensors() = 0;
        virtual void performAction(double p_action) = 0;
        virtual void reset() = 0;

    private:

    };
}


#endif //NEURONET_AREADEFENSEGAME_H
