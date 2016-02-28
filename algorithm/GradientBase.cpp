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

void GradientBase::calcDelta(NeuralGroup *p_group) {
  string id = p_group->getId();
  string outId;

  _delta[id] = VectorXd::Zero(p_group->getDim());
  for(vector<int>::iterator it = p_group->getOutConnections()->begin(); it != p_group->getOutConnections()->end(); it++) {
    outId = _network->getConnection(*it)->getOutGroup()->getId();
    for(int i = 0; i < _network->getConnection(*it)->getOutGroup()->getDim(); i++) {
      for(int j = 0; j < p_group->getDim(); j++) {
        _delta[id][j] += (*_network->getConnection(*it)->getWeights())(i, j) * _delta[outId][i];
      }
    }
  }

  for(int i = 0; i < p_group->getDim(); i++) {
    _delta[id][i] *= (*p_group->getDerivs())[i];
  }
}

void GradientBase::calcGradient(VectorXd* p_error) {
  _delta[_network->getOutputGroup()->getId()] = *p_error;

  for(auto it = _groupTree.begin(); it != _groupTree.end(); ++it) {
    (*it)->calcDerivs();
  }

  for(auto it = ++_groupTree.begin(); it != _groupTree.end(); ++it) {
    calcDelta(*it);
  }

  for(auto it = _network->getConnections()->rbegin(); it != _network->getConnections()->rend(); ++it) {
    calcGradient(it->second);
  }
}

void GradientBase::calcGradient(Connection *p_connection) {
  NeuralGroup* p_inGroup = p_connection->getInGroup();
  NeuralGroup* p_outGroup = p_connection->getOutGroup();

  _gradient[p_connection->getId()] = _delta[p_outGroup->getId()] * (*p_inGroup->getOutput()).transpose();
}
