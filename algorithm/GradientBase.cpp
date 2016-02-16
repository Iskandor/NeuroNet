//
// Created by Matej Pechac on 15. 2. 2016.
//

#include "GradientBase.h"

GradientBase::GradientBase(NeuralNetwork *p_network) {
  _network = p_network;

  bfsTreeCreate();
}

GradientBase::~GradientBase() {
}

void GradientBase::bfsTreeCreate() {
  for(auto it = _network->getGroups()->begin(); it != _network->getGroups()->end(); ++it) {
    it->second->invalidate();
  }
  _bfsTree.push_back(_network->getOutputGroup());
  bfsRecursive(_network->getOutputGroup());
}

void GradientBase::bfsRecursive(NeuralGroup* p_node) {
  p_node->setValid();
  for(vector<int>::iterator it = p_node->getInConnections()->begin(); it != p_node->getInConnections()->end(); ++it) {
    if (!_network->getConnections()->at(*it)->getInGroup()->isValid()) {
      _bfsTree.push_back(_network->getConnections()->at(*it)->getInGroup());
    }
  }

  for(vector<int>::iterator it = p_node->getInConnections()->begin(); it != p_node->getInConnections()->end(); ++it) {
    if (!_network->getConnections()->at(*it)->getInGroup()->isValid()) {
      bfsRecursive(_network->getConnections()->at(*it)->getInGroup());
    }
  }
}

void GradientBase::calcDelta(NeuralGroup *p_group) {
  int id = p_group->getId();
  int outId;

  _gradient[id] = VectorXd::Zero(p_group->getDim());
  for(vector<int>::iterator it = p_group->getOutConnections()->begin(); it != p_group->getOutConnections()->end(); it++) {
    outId = _network->getConnection(*it)->getOutGroup()->getId();
    for(int i = 0; i < _network->getConnection(*it)->getOutGroup()->getDim(); i++) {
      for(int j = 0; j < p_group->getDim(); j++) {
        _gradient[id][j] += (*_network->getConnection(*it)->getWeights())(i, j) * _gradient[outId][i];
      }
    }
  }
}

void GradientBase::calcGradient() {
  for(auto it = ++_bfsTree.begin(); it != _bfsTree.end(); ++it) {
    calcDelta(*it);
  }

  for(auto it = _bfsTree.rbegin(); it != _bfsTree.rend(); ++it) {
    calcGradient(*it);
  }
}

void GradientBase::calcGradient(NeuralGroup *p_group) {
  int id = p_group->getId();
  p_group->calcDerivs();

  for(int j = 0; j < p_group->getDim(); j++) {
    _gradient[id][j] *= (*p_group->getDerivs())(j);
  }
}
