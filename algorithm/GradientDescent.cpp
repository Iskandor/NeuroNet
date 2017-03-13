//
// Created by Matej Pechac on 15. 2. 2016.
//

#include <iostream>
#include "GradientDescent.h"

using namespace NeuroNet;

GradientDescent::GradientDescent(NeuralNetwork *p_network) {
    _network = p_network;

    groupTreeCreate();

    int nRows;
    int nCols;

    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); ++it) {
        nRows = it->second->getOutGroup()->getDim();
        nCols = it->second->getInGroup()->getDim();
        _regGradient[it->second->getId()] = MatrixXd::Zero(nRows, nCols);
        _invFisherMatrix[it->second->getId()] = MatrixXd::Identity(nRows, nRows);
    }
}

GradientDescent::~GradientDescent() {
}

void GradientDescent::groupTreeCreate() {
  for(auto it = _network->getGroups()->begin(); it != _network->getGroups()->end(); ++it) {
    it->second->invalidate();
  }
  _groupTree.push_back(_network->getOutputGroup());
  bfsRecursive(_network->getOutputGroup());
}

void GradientDescent::bfsRecursive(NeuralGroup* p_node) {
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

map<int, MatrixXd>* GradientDescent::calcRegGradient(VectorXd *p_error) {
    for(auto it = _groupTree.begin(); it != _groupTree.end(); ++it) {
        (*it)->calcDerivs();
    }

    _delta[_network->getOutputGroup()->getId()] = *_network->getOutputGroup()->getDerivs() * *p_error;

    for(auto it = ++_groupTree.begin(); it != _groupTree.end(); ++it) {
        deltaKernel(*it);
    }

    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); ++it) {
        regGradientKernel(it->second);
    }

    return &_regGradient;
}

void GradientDescent::deltaKernel(NeuralGroup *p_group) {
    Connection* connection = _network->getConnection(p_group->getOutConnection());
    string id = p_group->getId();
    string outId = connection->getOutGroup()->getId();
    _delta[id] = (*p_group->getDerivs()) * (connection->getWeights()->transpose() * _delta[outId]);
}

void GradientDescent::regGradientKernel(Connection *p_connection) {
    _regGradient[p_connection->getId()] = _delta[p_connection->getOutGroup()->getId()] * p_connection->getInGroup()->getOutput()->transpose();
}

map<int, MatrixXd>* GradientDescent::calcNatGradient(double p_epsilon, VectorXd *p_error) {
    _epsilon = p_epsilon;
    calcRegGradient(p_error);
    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); ++it) {
        invFisherMatrixKernel(it->second);
    }

    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); ++it) {
        natGradientKernel(it->second);
    }

    return &_natGradient;
}


void GradientDescent::invFisherMatrixKernel(Connection *p_connection) {
  int connectionId = p_connection->getId();
  _invFisherMatrix[connectionId] = (1 + _epsilon) * _invFisherMatrix[connectionId] - _epsilon * _invFisherMatrix[connectionId] * _regGradient[connectionId] * _regGradient[connectionId].transpose() * _invFisherMatrix[connectionId];
}

void GradientDescent::natGradientKernel(Connection *p_connection) {
  _natGradient[p_connection->getId()] = _invFisherMatrix[p_connection->getId()] * _regGradient[p_connection->getId()];
}
