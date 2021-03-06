//
// Created by mpechac on 7. 3. 2017.
//

#ifndef NEURONET_MAZEACTION_H
#define NEURONET_MAZEACTION_H

#include <string>

using namespace std;

class MazeAction {
public:
    MazeAction(string p_id, int p_x, int p_y);
    ~MazeAction();

    inline int X() {
        return _x;
    }

    inline int Y() {
        return _y;
    }

    inline string Id() {
        return _id;
    }


private:
    string _id;
    int _x;
    int _y;
};


#endif //NEURONET_MAZEACTION_H
