//
// Created by Matej Pechac on 15. 2. 2016.
//

#include <iostream>
#include "GradientBase.h"

using namespace NeuroNet;

GradientBase::GradientBase(NeuralNetwork *p_network) {
  _network = p_network;

  groupTreeCreate();

  int nRows;
  for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); ++it) {
    nRows = it->second->getOutGroup()->getDim();
    _invFisherMatrix[it->second->getId()] = MatrixXd::Identity(nRows, nRows);
  }

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

void GradientBase::calcGradient(VectorXd *p_error) {
    VectorXd unitary(_network->getOutputGroup()->getDim());
    unitary.fill(1);

    for(auto it = _groupTree.begin(); it != _groupTree.end(); ++it) {
        (*it)->calcDerivs();
    }

    if (p_error != nullptr) {
        _delta[_network->getOutputGroup()->getId()] = *_network->getOutputGroup()->getDerivs() * *p_error;
    }
    else {
        _delta[_network->getOutputGroup()->getId()] = *_network->getOutputGroup()->getDerivs() * unitary;
    }

    for(auto it = ++_groupTree.begin(); it != _groupTree.end(); ++it) {
        deltaKernel(*it);
    }

    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); ++it) {
        gradientKernel(it->second);
    }
}

void GradientBase::deltaKernel(NeuralGroup *p_group) {
    Connection* connection = _network->getConnection(p_group->getOutConnection());
    string id = p_group->getId();
    string outId = connection->getOutGroup()->getId();

    _delta[id] = VectorXd::Zero(p_group->getDim());
    for(int i = 0; i < connection->getOutGroup()->getDim(); i++) {
        for(int j = 0; j < p_group->getDim(); j++) {
            _delta[id][j] += _delta[outId][i] * (*connection->getWeights())(i, j);
        }
    }

    _delta[id] = (*p_group->getDerivs()) * _delta[id];
}

void GradientBase::gradientKernel(Connection *p_connection) {
  _gradient[p_connection->getId()] = _delta[p_connection->getOutGroup()->getId()] * p_connection->getInGroup()->getOutput()->transpose();
}

void GradientBase::calcNatGradient(double p_epsilon, VectorXd *p_error) {
  _epsilon = p_epsilon;
  calcGradient(_network->getOutput());
  for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); ++it) {
    invFisherMatrixKernel(it->second);
  }

  calcGradient(p_error);
  for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); ++it) {
    natGradientKernel(it->second);
  }
}


void GradientBase::invFisherMatrixKernel(Connection *p_connection) {
  int connectionId = p_connection->getId();
  _invFisherMatrix[connectionId] = (1 + _epsilon) * _invFisherMatrix[connectionId] - _epsilon * _invFisherMatrix[connectionId] * _gradient[connectionId] * _gradient[connectionId].transpose() * _invFisherMatrix[connectionId];
}

void GradientBase::natGradientKernel(Connection *p_connection) {
  _natGradient[p_connection->getId()] = _invFisherMatrix[p_connection->getId()] * _gradient[p_connection->getId()];
}
