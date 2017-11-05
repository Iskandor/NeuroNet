//
// Created by user on 19. 9. 2017.
//

#ifndef NEURONET_GRAPH_H
#define NEURONET_GRAPH_H

#include <map>
#include <string>
#include "Vertex.h"

using namespace std;

namespace NeuroNet {

class Graph : public Vertex {
protected:
    Vertex*                 _root;
    map<string, Vertex*>    _vertexBuffer;
    map<string, Edge<Vertex>*>      _edgeBuffer;
    list<Vertex*>           _forward;
    list<Vertex*>           _backward;

    void AddVertex(Vertex* p_vertex);
    void AddEdge(Edge<Vertex>* p_edge);

    void BFS();

public:
    Graph(string _id = "");
    virtual ~Graph();
};

}

#endif //NEURONET_GRAPH_H
