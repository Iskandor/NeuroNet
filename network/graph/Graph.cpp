//
// Created by user on 19. 9. 2017.
//

#include <queue>
#include "Graph.h"

using namespace NeuroNet;

Graph::Graph(string _id) : Vertex(_id) {

}

Graph::~Graph() {
    for(auto it  = _vertexBuffer.begin(); it != _vertexBuffer.end(); it++) {
        delete it->second;
    }

    for(auto it  = _edgeBuffer.begin(); it != _edgeBuffer.end(); it++) {
        delete it->second;
    }
}

void Graph::AddVertex(Vertex *p_vertex) {
    _vertexBuffer[p_vertex->Id()] = p_vertex;
}

void Graph::AddEdge(Edge<Vertex> *p_edge) {
    _edgeBuffer[p_edge->Id()] = p_edge;
    BFS();
}

/*
input G
for all u ∈ V(G)
     let visited[u] = false
let Q be empty
let r be a node of depth zero
push(Q,r)
while Q is not empty
     v = pop(Q)
     if visited[v] = false
          visit v
          visited[v] = true
          for each w ∈ V(G) such that (v,w) ∈ E(G) and visited[w] = false
               push(Q,w)
 */

void Graph::BFS() {
    map<string, bool> visited;

    for(auto it  = _vertexBuffer.begin(); it != _vertexBuffer.end(); it++) {
        visited[it->first] = false;
    }

    queue<string> q;

    q.push(_root->Id());

    while(!q.empty()) {
        string v = q.front();
        q.pop();

        if (!visited[v]) {
            _forward.push_back(_vertexBuffer[v]);
            visited[v] = true;
            for(auto it = _vertexBuffer[v]->OutEdges()->begin(); it != _vertexBuffer[v]->OutEdges()->end(); it++) {
                string w = ((Vertex*)(*it)->OutVertex())->Id();
                if (!visited[w]) {
                    q.push(w);
                }
            }
        }
    }
}
