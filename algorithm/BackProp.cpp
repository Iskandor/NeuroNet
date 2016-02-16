#include <math.h>
#include "../network/NeuralNetwork.h"
#include "BackProp.h"
#include "../algebra/matrix2.h"


BackProp::BackProp(NeuralNetwork* p_network) : GradientBase(p_network) {
  _alpha = 0;
  _weightDecay = 0;
  _momentum = 0;
  _input = nullptr;

  for(unsigned int i = 0; i < p_network->getGroups()->size(); i++) {
    int id = p_network->getGroups()->at(i)->getId();
    _gradient[id].init(p_network->getGroups()->at(i)->getDim());
  }
}

BackProp::~BackProp(void)
{
}

double BackProp::train(double *p_input, double* p_target) {
    double mse = 0;

    _input = p_input;
    
    // forward activation phase
    _network->setInput(p_input);
    _network->onLoop();

    // calc MSE
    for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
      mse += pow(p_target[i] - _network->getOutput()->at(i), 2);
    }

    // backward training phase
    for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
      _gradient[_network->getOutputGroup()->getId()][i] = p_target[i] - _network->getOutput()->at(i);
    }
    backProp();

    return mse;
}

void BackProp::backProp() {
    calcGradient();
    for(auto it = _bfsTree.rbegin(); it != _bfsTree.rend(); ++it) {
      update(*it);
    }
}


void BackProp::update(NeuralGroup* p_node) {
    for(vector<int>::iterator it = p_node->getInConnections()->begin(); it != p_node->getInConnections()->end(); it++) {         
      updateWeights(_network->getConnections()->at(*it)->getInGroup(), p_node, _network->getConnections()->at(*it));
      if (_weightDecay != 0) weightDecay(_network->getConnections()->at(*it));
    }
}


void BackProp::updateWeights(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, Connection* p_connection) {
  int nCols = p_inGroup->getDim();
  int nRows = p_outGroup->getDim();
  matrix2<double> delta(nRows, nCols);

  for(int i = 0; i < nRows; i++) {
    for(int j  = 0; j < nCols; j++) {
      delta.set(i, j, _alpha * _gradient[p_outGroup->getId()][i] * p_inGroup->getOutput()->at(j));
    }
  }

  *p_connection->getWeights() += delta;
}

void BackProp::weightDecay(Connection* p_connection) const
{
  *p_connection->getWeights() *= (1 - _weightDecay);
}

void BackProp::setAlpha(double p_alpha) {
  _alpha = p_alpha;
}

void BackProp::setWeightDecay(double p_weightDecay) {
  _weightDecay = p_weightDecay;
}

void BackProp::setMomentum(double p_momentum)
{
  _momentum = p_momentum;
}
