#include "Connection.h"
#include <random>

Connection::Connection(int p_id, NeuralGroup* p_inGroup, NeuralGroup* p_outGroup)
{
    _id = p_id;
    _inGroup = p_inGroup;
    _outGroup = p_outGroup;
    if (p_inGroup != nullptr) {
        _inDim = p_inGroup->getDim();
    }
    else {
        _inDim = p_outGroup->getDim();
    }
    _outDim = p_outGroup->getDim();
    _weights = new matrix2<double>(_outDim, _inDim);
}

Connection::~Connection(void)
{
    delete _weights;
}

/* initialize weights where density is from interval 0,1 and also inhibition which is count of negative (inhibitory) weights */
void Connection::init(double p_density, double p_inhibition) const {
    for(int i = 0; i < _outDim; i++) {
      for(int j = 0; j < _inDim; j++) {
          if (static_cast<double>(rand()) / RAND_MAX < p_density) {
              _weights->set(i, j, static_cast<double>(rand()) / RAND_MAX);
              if (static_cast<double>(rand()) / RAND_MAX < p_inhibition) {
                  _weights->set(i, j, _weights->at(i, j) * -1);
              }
          }
          else {
              _weights->set(i, j, 0);
          }
      }
    }
}