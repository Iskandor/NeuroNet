#include <iostream>
#include "NeuralNetwork.h"

using namespace NeuroNet;

NeuralNetwork::NeuralNetwork(void)
{
    _running = true;
    _connectionId = 0;
    _groupId = 0;
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

    Connection* recConnection = nullptr;
    /* transport information through the recurrent connections*/
    for(auto it = _recConnections.begin(); it != _recConnections.end(); it++) {
        recConnection = it->second;
        VectorXd* signal = recConnection->getInGroup()->getOutput();
        if (signal != nullptr) {
            recConnection->getOutGroup()->processInput(*signal);
            recConnection->getOutGroup()->integrate(signal, recConnection->getWeights());
            recConnection->getOutGroup()->fire();
        }
    }

    /* prepare input signal and propagate it through the network */
    _inputGroup->processInput(_input);
    _inputGroup->integrate(&_input, &_inputWeights, false);
    activate(_inputGroup);
    _output = *_outputGroup->getOutput();
}

void NeuralNetwork::activate(NeuralGroup* p_node) {
    NeuralGroup* inGroup = nullptr;
    /* sum input from all groups */
    for(vector<int>::iterator it = p_node->getInConnections()->begin(); it != p_node->getInConnections()->end(); it++) {
        /* generate output if it is possible */
        inGroup = _connections[*it]->getInGroup();

        VectorXd* signal = inGroup->getOutput();
        if (signal != nullptr) {
            p_node->processInput(*signal);
            p_node->integrate(signal, _connections[*it]->getWeights());
        }
    }

    p_node->fire();
    /* send signal to synapsis and repeat it for not activated group to prevent infinite loops */
    if (p_node->getOutConnection() != -1) {
        if (!_connections[p_node->getOutConnection()]->getOutGroup()->isValid()) {
            activate(_connections[p_node->getOutConnection()]->getOutGroup());
        }
    }
}

NeuralGroup* NeuralNetwork::addLayer(string p_id, int p_dim, NeuralGroup::ACTIVATION p_activationFunction, GROUP_TYPE p_type) {
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

Connection* NeuralNetwork::addConnection(string p_inGroupId, string p_outGroupId, double p_limit, double p_density,double p_inhibition) {
    return addConnection(_groups[p_inGroupId], _groups[p_outGroupId], p_limit, p_density, p_inhibition);
}

Connection* NeuralNetwork::addConnection(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, double p_limit, double p_density, double p_inhibition) {
    Connection* connection = new Connection(_connectionId, p_inGroup, p_outGroup);

    connection->init(p_limit);
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

void NeuralNetwork::activate(VectorXd *p_input) {
    setInput(p_input);
    onLoop();
}

Connection *NeuralNetwork::addRecConnection(NeuralGroup *p_inGroup, NeuralGroup *p_outGroup) {
    Connection* connection = new Connection(_connectionId, p_inGroup, p_outGroup);

    MatrixXd* weights = new MatrixXd(p_outGroup->getDim(), p_inGroup->getDim());
    weights->setIdentity();
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
        VectorXd zero = VectorXd::Zero(recConnection->getOutGroup()->getDim());
        recConnection->getOutGroup()->integrate(&zero, recConnection->getWeights());
        recConnection->getOutGroup()->fire();
    }
}

json NeuralNetwork::getFileData() {
    return json({{"type", "feedforward"}, {"ingroup", _inputGroup->getId()}, {"outgroup", _outputGroup->getId()}});
}
