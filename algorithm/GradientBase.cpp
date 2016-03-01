//
// Created by Matej Pechac on 15. 2. 2016.
//

#include <iostream>
#include "GradientBase.h"

GradientBase::GradientBase(NeuralNetwork *p_network) {
  _network = p_network;

  groupTreeCreate();
}

GradientBase::~GradientBase() {
}

void GradientBase::groupTreeCreate() {
  for(auto it = _network->getGroups()->begin(); it != _network->getGroups()->end(); ++it) {
    it->second->invalidate();
  }
  _groupTree.push_back(_network->getOutputGroup());
  bfsRecursive(_network->getOutputGroup());
}

void GradientBase::bfsRecursive(NeuralGroup* p_node) {
  p_node->setValid();
  for(vector<int>::iterator it = p_node->getInConnections()->begin(); it != p_node->getInConnections()->end(); ++it) {
    if (!_network->getConnections()->at(*it)->getInGroup()->isValid()) {
      _groupTree.push_back(_network->getConnections()->at(*it)->getInGroup());
    }
  }

  for(vector<int>::iterator it = p_node->getInConnections()->begin(); it != p_node->getInConnections()->end(); ++it) {
    if (!_network->getConnections()->at(*it)->getInGroup()->isValid()) {
      bfsRecursive(_network->getConnections()->at(*it)->getInGroup());
    }
  }
}

void GradientBase::gradientKernel(NeuralGroup *p_group) {
  string id = p_group->getId();
  string outId;

  _gradient[id] = VectorXd::Zero(p_group->getDim());
  for(vector<int>::iterator it = p_group->getOutConnections()->begin(); it != p_group->getOutConnections()->end(); it++) {
    outId = _network->getConnection(*it)->getOutGroup()->getId();
    for(int i = 0; i < _network->getConnection(*it)->getOutGroup()->getDim(); i++) {
      for(int j = 0; j < p_group->getDim(); j++) {
        _gradient[id][j] += _gradient[outId][i] * (*_network->getConnection(*it)->getWeights())(i, j);
      }
    }
  }

  for(int i = 0; i < p_group->getDim(); i++) {
    _gradient[id][i] *= (*p_group->getDerivs())[i];
  }
}

void GradientBase::calcGradient(VectorXd* p_error) {
  for(auto it = _groupTree.begin(); it != _groupTree.end(); ++it) {
    (*it)->calcDerivs();
  }

  if (p_error != nullptr) {
    _gradient[_network->getOutputGroup()->getId()] = *p_error;
  }
  else {
    _gradient[_network->getOutputGroup()->getId()] = *_network->getOutputGroup()->getDerivs();
  }

  for(auto it = ++_groupTree.begin(); it != _groupTree.end(); ++it) {
    gradientKernel(*it);
  }
}
