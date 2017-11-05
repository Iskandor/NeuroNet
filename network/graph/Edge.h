//
// Created by user on 19. 9. 2017.
//

#ifndef NEURONET_EDGE_H
#define NEURONET_EDGE_H

#include <string>

using namespace std;

namespace NeuroNet {

template <class T>
class Edge {
protected:
    string _id;
    T* _inVertex;
    T* _outVertex;

public:
    Edge(T* p_inVertex, T* p_outVertex);
    virtual ~Edge();

    inline string Id() {return _id; };
    inline T* InVertex() { return _inVertex; };
    inline T* OutVertex() { return _outVertex; };
};

}
#endif //NEURONET_EDGE_H
