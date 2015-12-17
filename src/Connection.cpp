#include "Connection.h"
#include <random>

Connection::Connection(int p_id, NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, int p_speed)
{
    _id = p_id;
    _inGroup = p_inGroup;
    _outGroup = p_outGroup;
    if (p_inGroup != NULL) {
        _inDim = p_inGroup->getDim();
    }
    else {
        _inDim = p_outGroup->getDim();
    }
    _outDim = p_outGroup->getDim();
    _speed = p_speed;
    _weights = new matrix2<double>(_outDim, _inDim);
}

Connection::~Connection(void)
{
    delete _weights;
}

/* initialize weights where density is from interval 0,1 and also inhibition which is count of negative (inhibitory) weights */
void Connection::init(double p_density, double p_inhibition) {
    for(int i = 0; i < _outDim; i++) {
      for(int j = 0; j < _inDim; j++) {
          if (((double) rand() / (RAND_MAX)) < p_density) {
              _weights->set(i, j, ((double) rand() / (RAND_MAX)));
              if (((double) rand() / (RAND_MAX)) < p_inhibition) {
                  _weights->set(i, j, _weights->at(j, i) * -1);
              }
          }
          else {
              _weights->set(i, j, 0);
          }
      }
    }
}

/* adds output of input group to connection pipeline */
void Connection::addSignal(double* p_signal) {
    double* signal = new double[_inDim];
    memcpy(signal, p_signal, sizeof(double)*_inDim);
    _signals.push_back(pair<int, double*>(0, signal));
}

/* pushes forward the signals in pipeline */
void Connection::loopSignal() {
    for(vector<pair<int, double*>>::iterator it = _signals.begin(); it != _signals.end(); it++) {
        (*it).first++;
    }
}

/* gets signal which is in the end of the pipeline (its position is equal to speed parameter) */
double* Connection::getSignal() {
    if (!_signals.empty() && (*_signals.begin()).first == _speed) {
        double* signal = (*_signals.begin()).second;
        _signals.erase(_signals.begin());
        return signal;
    }

    return NULL;
}