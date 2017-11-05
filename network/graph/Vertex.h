//
// Created by user on 19. 9. 2017.
//

#ifndef NEURONET_VERTEX_H
#define NEURONET_VERTEX_H

#include <string>
#include <list>
#include "Edge.h"

using namespace std;

namespace NeuroNet {

class Vertex {
protected:
    string _id;
    list<Edge<Vertex>*> _inEdges;
    list<Edge<Vertex>*> _outEdges;

public:
    Vertex(string p_id);
    virtual ~Vertex();

    inline string Id() {return _id; };
    inline list<Edge<Vertex>*>* InEdges() {return &_inEdges; };
    inline list<Edge<Vertex>*>* OutEdges() {return &_outEdges; };
};

}

#endif //NEURONET_VERTEX_H
