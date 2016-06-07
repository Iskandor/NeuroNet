#include <random>
#include "Connection.h"
#include "NetworkUtils.h"
#include "RandomGenerator.h"

using namespace NeuroNet;

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
    _weights = new MatrixXd(_outDim, _inDim);
}

Connection::~Connection(void)
{
    delete _weights;
}

/* initialize weights where density is from interval 0,1 and also inhibition which is count of negative (inhibitory) weights */
void Connection::init(double p_density, double p_inhibition) const {
    RandomGenerator generator;
    for(int i = 0; i < _outDim; i++) {
      for(int j = 0; j < _inDim; j++) {
          if (generator.random() < p_density) {
              (*_weights)(i, j) =  generator.random() * 0.1;
              if (generator.random() < p_inhibition) {
                  (*_weights)(i, j) *= -1;
              }
          }
          else {
              (*_weights)(i, j) = 0;
          }
      }
    }
}

void Connection::init(MatrixXd *p_weights) {
    _weights = p_weights;
}

json Connection::getFileData() {
    string weights;

    for(int i = 0; i < _outDim; i++) {
        for (int j = 0; j < _inDim; j++) {
            weights += to_string((*_weights)(i, j));
            weights += "|";
        }
    }

    return json({{"ingroup", _inGroup->getId()}, {"outgroup", _outGroup->getId()}, {"weights", weights}});
}
