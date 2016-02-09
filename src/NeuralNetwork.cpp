#include "NeuralNetwork.h"
#include <iostream>

NeuralNetwork::NeuralNetwork(void)
{
    _running = true;
    _groupId = 0;
    _connectionId = 0;
}

NeuralNetwork::~NeuralNetwork(void)
{
    for(map<int, NeuralGroup*>::iterator it = _groups.begin(); it != _groups.end(); it++) {
        delete it->second;
    }
    for(map<int, Connection*>::iterator it = _connections.begin(); it != _connections.end(); it++) {
        delete it->second;
    }
}

void NeuralNetwork::onLoop() {
    /* invalidate all neural groups */
    for(map<int, NeuralGroup*>::iterator it = _groups.begin(); it != _groups.end(); it++) {
        it->second->invalidate();
    }
    /* prepare input signal and propagate it through the network */
    _inputGroup->integrate(&_input, &_inputWeights);
    activate(_inputGroup);
    _output.setVector(_outputGroup->getOutput());
}

void NeuralNetwork::activate(NeuralGroup* p_node) {
    NeuralGroup* inGroup;
    /* sum input from all groups */
    for(vector<int>::iterator it = p_node->getInConnections()->begin(); it != p_node->getInConnections()->end(); it++) {
        /* generate output if it is possible */
        inGroup = _connections[*it]->getInGroup();        

        vectorN<double>* signal = inGroup->getOutput();
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

NeuralGroup* NeuralNetwork::addLayer(int p_dim, int p_activationFunction, GROUP_TYPE p_type) {
    NeuralGroup* group = new NeuralGroup(_groupId, p_dim, p_activationFunction);
    _groups[_groupId] = group;
    _groupId++;
    switch(p_type) {
        case INPUT:
            _inputGroup = group;
            _input.init(group->getDim());
            /* initialize input weights to unitary matrix */
            _inputWeights.init(group->getDim(), group->getDim(), matrix2<double>::UNITARY);
        break;

        case OUTPUT:
            _outputGroup = group;
            _output.init(group->getDim());
        break;
        case HIDDEN: 
        break;
        default: 
        break;
    }

    return group;
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
