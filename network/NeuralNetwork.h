#pragma once
#include "../algebra/vectorN.h"
#include "NeuralGroup.h"
#include "Connection.h"
#include "../algebra/matrix2.h"
#include <map>

using namespace std;

class NeuralNetwork
{
public:
    enum GROUP_TYPE {
        HIDDEN = 0,
        INPUT = 1,
        OUTPUT = 2
    };

	NeuralNetwork(void);
	~NeuralNetwork(void);

  void onLoop();
  NeuralGroup* addLayer(int p_dim, int p_activationFunction, GROUP_TYPE p_type);
  Connection* addConnection(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, double p_density = 1, double p_inhibition = 0.5);

  bool running() const { return _running; };
  vectorN<double>* getOutput() { return &_output; };
  double getScalarOutput() const { return _output[0]; };
  map<int, NeuralGroup*>* getGroups() { return &_groups; };
  NeuralGroup* getOutputGroup() const { return _outputGroup; };
  NeuralGroup* getInputGroup() const { return _inputGroup; };
  map<int, Connection*>* getConnections() { return &_connections; };
  Connection* getConnection(int p_id) { return _connections[p_id]; };
  NeuralGroup* getGroup(int p_id) { return _groups[p_id];};

  void setInput(vectorN<double> *p_input) { _input = *p_input; };
  void setInput(double *p_input) { _input.setVector(p_input); };

private:
    void activate(NeuralGroup* p_node);

    int _groupId;
    int _connectionId;

    NeuralGroup* _inputGroup;
    NeuralGroup* _outputGroup;

    map<int, NeuralGroup*> _groups;
    map<int, Connection*> _connections;

    matrix2<double> _inputWeights;
    vectorN<double> _input;
    vectorN<double> _output;

    bool _running;
};

