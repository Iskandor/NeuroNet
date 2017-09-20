#include <random>
#include "Connection.h"
#include "NetworkUtils.h"
#include "../backend/FLAB/RandomGenerator.h"

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
    _weights = new Matrix(_outDim, _inDim);
}

Connection::Connection(Connection &p_copy) {
    _id = p_copy._id;
    _inGroup = p_copy._inGroup;
    _outGroup = p_copy._outGroup;
    _inDim = p_copy._inDim;
    _outDim = p_copy._outDim;
    _weights = new Matrix(*p_copy._weights);
}

Connection::~Connection(void)
{
    delete _weights;
}

/* initialize weights where density is from interval 0,1 and also inhibition which is count of negative (inhibitory) weights */
void Connection::init(double p_density, double p_inhibition) const {
    for(int i = 0; i < _outDim; i++) {
      for(int j = 0; j < _inDim; j++) {
          if (RandomGenerator::getInstance().random() < p_density) {
              _weights->set(i, j, RandomGenerator::getInstance().random() * 0.1);
              if (RandomGenerator::getInstance().random() < p_inhibition) {
                  _weights->set(i, j, _weights->at(i,j) * -1);
              }
          }
          else {
              _weights->set(i, j, 0);
          }
      }
    }
}

void Connection::init(Connection::INIT p_init, double p_limit) {
    switch(p_init) {
        case UNIFORM:
            uniform(p_limit);
            break;
        case LECUN_UNIFORM:
            uniform((double) pow(_inDim, -.5));
            break;
        case GLOROT_UNIFORM:
            uniform(2 / (_inDim + _outDim));
            break;
        case IDENTITY:
            identity();
            break;
    }
}

void Connection::init(Matrix *p_weights) {
    _weights = p_weights;
}

json Connection::getFileData() {
    string weights;

    for(int i = 0; i < _outDim; i++) {
        for (int j = 0; j < _inDim; j++) {
            weights += to_string(_weights->at(i,j));
            weights += "|";
        }
    }

    return json({{"ingroup", _inGroup->getId()}, {"outgroup", _outGroup->getId()}, {"weights", weights}});
}

void Connection::uniform(double p_limit) {
    for(int i = 0; i < _outDim; i++) {
        for (int j = 0; j < _inDim; j++) {
            _weights->set(i, j, RandomGenerator::getInstance().random(-p_limit, p_limit));
            //(*_weights)[i][j] = 0;
        }
    }
}

void Connection::identity() {
    _weights = new Matrix(_outDim, _inDim, Matrix::IDENTITY);
}

void Connection::setWeights(Matrix *p_weights) {
    delete _weights;
    _weights = new Matrix(*p_weights);
}
