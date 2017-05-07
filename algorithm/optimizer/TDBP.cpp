//
// Created by mpechac on 28. 3. 2017.
//

#include "TDBP.h"

using namespace NeuroNet;

TDBP::TDBP(NeuralNetwork *p_network, double p_lambda, double p_weightDecay, double p_momentum, bool p_nesterov, const GradientDescent::GRADIENT &p_gradient) : BackProp(p_network, p_weightDecay, p_momentum, p_nesterov, p_gradient) {
    int nRows;
    int nCols;

    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); ++it) {
        int x = it->second->getInGroup()->getDim();
        int y = it->second->getOutGroup()->getDim();
        int z = _network->getOutput()->size();
        _e[it->second->getId()] = Tensor3::Zero(x, y, z);
    }

    for(auto it = _network->getGroups()->begin(); it != _network->getGroups()->end(); ++it) {
        _d[it->second->getId()] = Matrix::Zero(_network->getOutput()->size(), it->second->getDim());
    }
}

TDBP::~TDBP(void) {

}

double TDBP::train(Vector *p_input, Vector *p_target) {
    double mse = 0;

    // forward activation phase
    _network->activate(p_input);

    // backward training phase
    _error = (*p_target) - (*_network->getOutput());

    mse = calcMse(p_target);
    //calcGradient(&_error);
    calcEligTrace();

    for(auto it = _groupTree.rbegin(); it != _groupTree.rend(); ++it) {
        update(*it);
    }

    return mse;
}

void TDBP::updateWeights(Connection *p_connection) {
    int id = p_connection->getId();

    for(int j = 0; j < p_connection->getOutGroup()->getDim(); j++) {
        for(int k = 0; k < p_connection->getInGroup()->getDim(); k++) {
            double s = 0;
            for(int i = 0; i < _network->getOutput()->size(); i++) {
                s += _error[i] * _e[id](k, j, i);
            }
            (*p_connection->getWeights())[j][k] += s;
        }
    }
}

map<int, Tensor3> *TDBP::calcEligTrace() {
    for(auto it = _groupTree.begin(); it != _groupTree.end(); ++it) {
        (*it)->calcDerivs();
    }

    string id = _network->getOutputGroup()->getId();
    for(int i = 0; i < _network->getOutputGroup()->getDim(); i++) {
        (_d[id])[i][i] = *_network->getOutputGroup()->getDerivs()[i][i];
    }

    for(auto it = ++_groupTree.begin(); it != _groupTree.end(); ++it) {
        NeuralGroup* group = *it;
        Connection* connection = _network->getConnection(group->getOutConnection());
        string oid = connection->getOutGroup()->getId();
        string iid = connection->getInGroup()->getId();

        _d[iid] = _d[oid] * *connection->getWeights() * *_network->getGroup(iid)->getDerivs();
    }

    /* je tu memory leak */
    for(auto it = _network->getConnections()->begin(); it != _network->getConnections()->end(); ++it) {
        Connection* connection = it->second;
        int cid = connection->getId();
        string oid = connection->getOutGroup()->getId();
        string iid = connection->getInGroup()->getId();

        for(int i = 0; i < _network->getOutput()->size(); i++) {
            for(int j = 0; j < connection->getOutGroup()->getDim(); j++) {
                for (int k = 0; k < connection->getInGroup()->getDim(); k++) {
                    double val = _d[oid][i][j] * (*_network->getGroup(iid)->getOutput())[k];
                    _e[cid].set(k, j, i, val);
                }
            }
        }
    }

    return &_e;
}
