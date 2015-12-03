#include "NeuralNetwork.h"
#include <iostream>

NeuralNetwork::NeuralNetwork(void)
{
    _running = true;
    _groupId = 0;
}

NeuralNetwork::~NeuralNetwork(void)
{
    if (_inputWeights != NULL) delete[] _inputWeights;

    for(vector<NeuralGroup*>::iterator it = _groups.begin(); it != _groups.end(); it++) {
        delete *it;
    }
    for(vector<Connection*>::iterator it = _connections.begin(); it != _connections.end(); it++) {
        delete *it;
    }
}

void NeuralNetwork::init() {
    for(vector<NeuralGroup*>::iterator it = _groups.begin(); it != _groups.end(); it++) {
        ((NeuralGroup*)(*it))->init();
    }
}

void NeuralNetwork::onLoop() {
    /* invalidate all neural groups */
    for(vector<NeuralGroup*>::iterator it = _groups.begin(); it != _groups.end(); it++) {
        ((NeuralGroup*)(*it))->invalidate();
    }
    /* push all signals in synapses towards their destination groups */
    for(vector<Connection*>::iterator it = _connections.begin(); it != _connections.end(); it++) {
        ((Connection*)(*it))->loopSignal();
    }
    /* prepare input signal and propagate it through the network */
    _inputGroup->integrate(_input, _inputWeights, _inputGroup->getDim());
    activate(_inputGroup);
    _output = _outputGroup->getOutput();


/*
    for(int i = 0; i < _outputGroup->getDim(); i++) {
        cout << _output[i] << " ";
    }
    cout << endl;
*/
}

void NeuralNetwork::activate(NeuralGroup* p_node) {
    NeuralGroup* inGroup;
    /* sum input from all groups */
    for(vector<int>::iterator it = p_node->getInConnections()->begin(); it != p_node->getInConnections()->end(); it++) {
        /* generate output if it is possible */
        inGroup = _connections[*it]->getInGroup();        

        double* signal = inGroup->getOutput();
        if (signal != NULL) {
            p_node->integrate(signal, _connections[*it]->getWeights()->getMatrix(), _connections[*it]->getInGroup()->getDim());
            //delete[] signal;
        }
    }

    p_node->fire();
    /* send signal to synapsis and repeat it for not activated group to prevent infinite loops */
    for(vector<int>::iterator it = p_node->getOutConnections()->begin(); it != p_node->getOutConnections()->end(); it++) {        
        if (!_connections[*it]->getOutGroup()->isActivated()) {
            activate(_connections[*it]->getOutGroup());
        }
    }
}

NeuralGroup* NeuralNetwork::addLayer(int p_dim, int p_activationFunction, GROUP_TYPE p_type) {
    NeuralGroup* group = new NeuralGroup(_groupId, p_dim, p_activationFunction);
    _groupId++;

    _groups.push_back(group);
    switch(p_type) {
        case INPUT:
            _inputGroup = group;
            _input = new double[group->getDim()];
            /* initialize input weights to unitary matrix */
            _inputWeights = new double[group->getDim() * group->getDim()];
            memset(_inputWeights, 0, sizeof(double)*group->getDim() * group->getDim());

            for(int i = 0; i < group->getDim(); i++) {
                _inputWeights[i*group->getDim()+i] = 1;
            }
        break;

        case OUTPUT:
            _outputGroup = group;
            _output = new double[group->getDim()];
        break;
    }

    return group;
}

Connection* NeuralNetwork::addConnection(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, double p_density, double p_inhibition, int p_speed) {
    Connection* connection = new Connection(p_inGroup, p_outGroup, p_speed);

    connection->init(p_density, p_inhibition);
    _connections.push_back(connection);
    if (p_inGroup != NULL) p_inGroup->addOutConnection(_connections.size() - 1);
    if (p_outGroup != NULL) p_outGroup->addInConnection(_connections.size() - 1);

    return connection;
}
