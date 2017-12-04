//
// Created by user on 5. 11. 2017.
//

#ifndef NEURONET_SEQUENCE_H
#define NEURONET_SEQUENCE_H

#include <vector>
#include <Vector.h>
#include <map>

using namespace std;
using namespace FLAB;

namespace MNS {

class Sequence {
public:
    Sequence(int p_id, int p_grasp);
    ~Sequence();

    void addMotorData(Vector *p_data);
    void addVisualData(int p_perspective, Vector *p_data);

    vector<Vector*>* getMotorData();
    vector<Vector*>* getVisualData();
    vector<Vector*>* getVisualData(int p_perspective);

    inline int getGrasp() {return _grasp;};

private:
    int _id;
    int _grasp;


    map<int, vector<Vector*>> _v_buffer;
    vector<Vector*> _v_data;
    vector<Vector*> _m_data;
};

}



#endif //NEURONET_SEQUENCE_H
