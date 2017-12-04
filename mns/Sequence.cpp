//
// Created by user on 5. 11. 2017.
//

#include "Sequence.h"

using namespace MNS;

Sequence::Sequence(int p_id, int p_grasp) {
    _id = p_id;
    _grasp = p_grasp;
}

Sequence::~Sequence() {
    for(auto it = _m_data.begin(); it != _m_data.end(); it++) {
        delete *it;
    }

    for(auto it = _v_data.begin(); it != _v_data.end(); it++) {
        delete *it;
    }
}

void Sequence::addMotorData(Vector *p_data) {
    _m_data.push_back(p_data);
}

void Sequence::addVisualData(int p_perspective, Vector *p_data) {
    _v_buffer[p_perspective].push_back(p_data);
    _v_data.push_back(p_data);
}

vector<Vector *> *Sequence::getMotorData() {
    return &_m_data;
}

vector<Vector *> *Sequence::getVisualData() {
    return &_v_data;
}

vector<Vector *> *Sequence::getVisualData(int p_perspective) {
    return &_v_buffer[p_perspective];
}
