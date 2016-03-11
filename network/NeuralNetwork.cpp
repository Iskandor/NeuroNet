#include <iostream>
#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(void)
{
    _running = true;
    _connectionId = 0;
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
    /* prepare input signal and propagate it through the network */
    _inputGroup->integrate(&_input, &_inputWeights);
    activate(_inputGroup);
    _output = *_outputGroup->getOutput();
}

void NeuralNetwork::activate(NeuralGroup* p_node) {
    NeuralGroup* inGroup;
    /* sum input from all groups */
    for(vector<int>::iterator it = p_node->getInConnections()->begin(); it != p_node->getInConnections()->end(); it++) {
        /* generate output if it is possible */
        inGroup = _connections[*it]->getInGroup();

        VectorXd* signal = inGroup->getOutput();
        if (signal != nullptr) {
            p_node->integrate(signal, _connections[*it]->getWeights());
        }
    }

    p_node->fire();
    /* send signal to synapsis and repeat it for not activated group to prevent infinite loops */
    for(vector<int>::iterator it = p_node->getOutConnections()->begin(); it != p_node->getOutConnections()->end(); it++) {        
        if (!_connections[*it]->getOutGroup()->isValid()) {
            activate(_connections[*it]->getOutGroup());
        }
    }
}

NeuralGroup* NeuralNetwork::addLayer(string p_id, int p_dim, int p_activationFunction, GROUP_TYPE p_type) {
    NeuralGroup* group = new NeuralGroup(p_id, p_dim, p_activationFunction);
    _groups[p_id] = group;
    _groupId++;
    switch(p_type) {
        case INPUT:
            _inputGroup = group;
            _input.resize(group->getDim());
            /* initialize input weights to unitary matrix */
            _inputWeights = MatrixXd::Identity(group->getDim(), group->getDim());
        break;

        case OUTPUT:
            _outputGroup = group;
            _output.resize(group->getDim());
        break;
        case HIDDEN: 
        break;
        default: 
        break;
    }

    return group;
}

Connection* NeuralNetwork::addConnection(string p_inGroupId, string p_outGroupId, double p_density,double p_inhibition) {
    addConnection(_groups[p_inGroupId], _groups[p_outGroupId], p_density, p_inhibition);
}

Connection* NeuralNetwork::addConnection(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, double p_density, double p_inhibition) {
    Connection* connection = new Connection(_connectionId, p_inGroup, p_outGroup);    

    connection->init(p_density, p_inhibition);
    _connections[_connectionId] = connection;
    if (p_inGroup != nullptr) p_inGroup->addOutConnection(_connectionId);
    if (p_outGroup != nullptr) p_outGroup->addInConnection(_connectionId);
    _connectionId++;

    return connection;
}

void NeuralNetwork::setInput(double *p_input) {
  for(int i = 0; i < _input.rows(); i++) {
    _input[i] = p_input[i];
  }
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
