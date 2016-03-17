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
  Connection* connection = _network->getConnection(p_group->getOutConnection());
  string id = p_group->getId();
  string outId = connection->getOutGroup()->getId();

  _delta[id] = VectorXd::Zero(p_group->getDim());
  for(int i = 0; i < connection->getOutGroup()->getDim(); i++) {
    for(int j = 0; j < p_group->getDim(); j++) {
      _delta[id][j] += _delta[outId][i] * (*connection->getWeights())(i, j);
    }
  }

  for(int i = 0; i < p_group->getDim(); i++) {
    _delta[id][i] *= (*p_group->getDerivs())[i];
  }
}

void GradientBase::calcGradient(VectorXd *p_error) {
  for(auto it = _groupTree.begin(); it != _groupTree.end(); ++it) {
    (*it)->calcDerivs();
  }

  if (p_error != nullptr) {
    _delta[_network->getOutputGroup()->getId()] = p_error->cwiseProduct(*_network->getOutputGroup()->getDerivs());
  }
  else {
    _delta[_network->getOutputGroup()->getId()] = *_network->getOutputGroup()->getDerivs();
  }

  for(auto it = ++_groupTree.begin(); it != _groupTree.end(); ++it) {
    gradientKernel(*it);
  }
}
