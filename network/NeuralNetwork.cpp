#include <iostream>
#include "NeuralNetwork.h"

using namespace NeuroNet;

NeuralNetwork::NeuralNetwork(void)
{
    _connectionId = 0;
}

NeuralNetwork::NeuralNetwork(NeuralNetwork &p_copy) {
    for (auto it = p_copy._groups.begin(); it != p_copy._groups.end(); it++) {
        NeuralGroup g(*it->second);

        if (p_copy._inputGroup == it->second) {
            addLayer(&g, INPUT);
        }
        else if (p_copy._outputGroup == it->second) {
            addLayer(&g, OUTPUT);
        }
        else {
            addLayer(&g, HIDDEN);
        }
    }

    for (auto it = p_copy._connections.begin(); it != p_copy._connections.end(); it++) {
        Connection c(*it->second);
        addConnection(&c);
    }

    _connectionId = p_copy._connectionId;
}

NeuralNetwork::~NeuralNetwork(void)
{
    for(map<string, NeuralGroup*>::iterator it = _groups.begin(); it != _groups.end(); it++) {
        delete it->second;
    }
    for(map<int, Connection*>::iterator it = _connections.begin(); it != _connections.end(); it++) {
        delete it->second;
    }
}

void NeuralNetwork::onLoop() {
    /* invalidate all neural groups */
    for(map<string, NeuralGroup*>::iterator it = _groups.begin(); it != _groups.end(); it++) {
        it->second->invalidate();
    }

    /*
    Connection* recConnection = nullptr;
    /* transport information through the recurrent connections
    for(auto it = _recConnections.begin(); it != _recConnections.end(); it++) {
        recConnection = it->second;
        Vector* signal = recConnection->getInGroup()->getOutput();
        if (signal != nullptr) {
            recConnection->getOutGroup()->processInput(*signal);
            recConnection->getOutGroup()->integrate(signal, recConnection->getWeights());
            recConnection->getOutGroup()->fire();
        }
    }
    */

    /* prepare input signal and propagate it through the network */
    activate(_inputGroup);
    _output = *_outputGroup->getOutput();
}

void NeuralNetwork::activate(NeuralGroup* p_node) {
    NeuralGroup* inGroup = nullptr;
    /* sum input from all groups */
    for(vector<int>::iterator it = p_node->getInConnections()->begin(); it != p_node->getInConnections()->end(); it++) {
        /* generate output if it is possible */
        inGroup = _connections[*it]->getInGroup();

        Vector* signal = inGroup->getOutput();
        if (signal != nullptr) {
            //p_node->processInput(*signal);
            p_node->integrate(signal, _connections[*it]->getWeights());
        }
    }

    p_node->fire();
    /* send signal to synapsis and repeat it for not activated group to prevent infinite loops */
    for(vector<int>::iterator it = p_node->getOutConnection()->begin(); it != p_node->getOutConnection()->end(); it++) {
        if (!_connections[*it]->getOutGroup()->isValid()) {
            activate(_connections[*it]->getOutGroup());
        }
    }
}

NeuralGroup* NeuralNetwork::addLayer(string p_id, int p_dim, NeuralGroup::ACTIVATION p_activationFunction, GROUP_TYPE p_type, bool p_bias) {
    bool bias = p_type == INPUT ? false : p_bias;
    NeuralGroup* group = new NeuralGroup(p_id, p_dim, p_activationFunction, bias);
    _groups[p_id] = group;
    switch(p_type) {
        case INPUT:
            _inputGroup = group;
            break;

        case OUTPUT:
            _outputGroup = group;
            _output = Vector::Zero(group->getDim());
            break;
        case HIDDEN:
            break;
        default:
            break;
    }

    return group;
}

NeuralGroup *NeuralNetwork::addLayer(NeuralGroup *p_group, GROUP_TYPE p_type) {
    _groups[p_group->getId()] = p_group;

    switch(p_type) {
        case INPUT:
            _inputGroup = p_group;
            break;

        case OUTPUT:
            _outputGroup = p_group;
            _output = Vector::Zero(p_group->getDim());
            break;
        case HIDDEN:
            break;
        default:
            break;
    }

    return p_group;
}

Connection* NeuralNetwork::addConnection(string p_inGroupId, string p_outGroupId, Connection::INIT p_init, double p_limit) {
    return addConnection(_groups[p_inGroupId], _groups[p_outGroupId], p_init, p_limit);
}

Connection* NeuralNetwork::addConnection(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, Connection::INIT p_init, double p_limit) {
    Connection* connection = new Connection(_connectionId, p_inGroup, p_outGroup);

    connection->init(p_init, p_limit);
    _connections[_connectionId] = connection;
    if (p_inGroup != nullptr) p_inGroup->addOutConnection(_connectionId);
    if (p_outGroup != nullptr) p_outGroup->addInConnection(_connectionId);
    _connectionId++;

    return connection;
}

Connection *NeuralNetwork::addConnection(Connection *p_connection) {
    _connections[p_connection->getId()] = p_connection;
    return p_connection;
}

Connection* NeuralNetwork::getConnection(string p_inGroupId, string p_outGroupId) {
    Connection* result = nullptr;
    for(auto it = _connections.begin(); it != _connections.end(); it++) {
        if (it->second->getInGroup()->getId() == p_inGroupId && it->second->getOutGroup()->getId() == p_outGroupId) {
            result = it->second;
        }
    }
    return result;
}

void NeuralNetwork::activate(Vector *p_input) {
    setInput(p_input);
    onLoop();
}

Connection *NeuralNetwork::addRecConnection(NeuralGroup *p_inGroup, NeuralGroup *p_outGroup) {
    Connection* connection = new Connection(_connectionId, p_inGroup, p_outGroup);

    Matrix* weights = new Matrix(p_outGroup->getDim(), p_inGroup->getDim(), Matrix::IDENTITY);
    connection->init(weights);
    _recConnections[_connectionId] = connection;
    _connectionId++;

    return connection;
}

Connection *NeuralNetwork::addRecConnection(string p_inGroupId, string p_outGroupId) {
    return addRecConnection(_groups[p_inGroupId], _groups[p_outGroupId]);
}

void NeuralNetwork::resetContext() {
    Connection* recConnection;
    for(auto it = _recConnections.begin(); it != _recConnections.end(); it++) {
        recConnection = it->second;
        Vector zero = Vector::Zero(recConnection->getOutGroup()->getDim());
        recConnection->getOutGroup()->integrate(&zero, recConnection->getWeights());
        recConnection->getOutGroup()->fire();
    }
}

json NeuralNetwork::getFileData() {
    return json({{"type", "feedforward"}, {"ingroup", _inputGroup->getId()}, {"outgroup", _outputGroup->getId()}});
}

void NeuralNetwork::overrideParams(NeuralNetwork* p_source) {
    for (auto it = p_source->_groups.begin(); it != p_source->_groups.end(); it++) {
        _groups[it->first]->setBias(it->second->getBias());
    }

    for (auto it = p_source->_connections.begin(); it != p_source->_connections.end(); it++) {
        _connections[it->first]->setWeights(it->second->getWeights());
    }
}

void NeuralNetwork::setInput(Vector *p_input) {
    _inputGroup->setOutput(p_input);
}
