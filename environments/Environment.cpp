//
// Created by mpechac on 7. 3. 2017.
//

#include "Environment.h"

using namespace NeuroNet;

Environment::Environment() {
    _indim = _outdim = 0;
    _discreteStates = _discreteActions = false;
    _numActions = 0;
}

Environment::~Environment() {

}
