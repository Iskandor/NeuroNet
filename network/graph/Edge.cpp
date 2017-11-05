//
// Created by user on 19. 9. 2017.
//

#include <random>
#include "Edge.h"

using namespace NeuroNet;

template <class T>
Edge<T>::Edge(T *p_inVertex, T *p_outVertex) {
    std::random_device _rd;
    std::mt19937 _mt;
    _mt.seed(_rd());

    uniform_int_distribution<int> distribution;

    _id = distribution(_mt);
    _inVertex = p_inVertex;
    _outVertex = p_outVertex;
}

template <class T>
Edge<T>::~Edge() {

}
