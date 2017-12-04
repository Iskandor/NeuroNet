#pragma once
#include "NeuralGroup.h"
#include "Connection.h"
#include <map>

using namespace std;
using json = nlohmann::json;

namespace NeuroNet {

    class NeuralNetwork
    {
    public:
        enum GROUP_TYPE {
            HIDDEN = 0,
            INPUT = 1,
            OUTPUT = 2
        };

        NeuralNetwork(void);
        NeuralNetwork(NeuralNetwork& p_copy);
        virtual ~NeuralNetwork(void);

        NeuralGroup* addLayer(string p_id, int p_dim, NeuralGroup::ACTIVATION p_activationFunction, GROUP_TYPE p_type, bool p_bias = true);
        NeuralGroup* addLayer(NeuralGroup* p_group, GROUP_TYPE p_type);

        Connection* addConnection(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup, Connection::INIT p_init = Connection::UNIFORM, double p_limit = 0.05);
        Connection* addConnection(string p_inGroupId, string p_outGroupId, Connection::INIT p_init = Connection::UNIFORM, double p_limit = 0.05);
        Connection* addConnection(Connection* p_connection);
        Connection* addRecConnection(NeuralGroup* p_inGroup, NeuralGroup* p_outGroup);
        Connection* addRecConnection(string p_inGroupId, string p_outGroupId);

        void overrideParams(NeuralNetwork* p_source);

        Vector* getOutput() { return &_output; };
        //double getScalarOutput() const { return _output[0]; };

        map<string, NeuralGroup*>* getGroups() { return &_groups; };
        map<int, Connection*>* getConnections() { return &_connections; };
        Connection* getConnection(int p_id) { return _connections[p_id]; };
        Connection* getConnection(string p_inGroupId, string p_outGroupId);
        map<int, Connection*>* getRecConnections() { return &_recConnections; };
        NeuralGroup* getGroup(string p_id) { return _groups[p_id];};
        NeuralGroup* getOutputGroup() { return _outputGroup;};

        void setInput(Vector *p_input);
        void onLoop();
        virtual void resetContext();
        virtual void activate(Vector *p_input);

        virtual json getFileData();

    protected:
        void activate(NeuralGroup* p_node);

        int _connectionId;

        NeuralGroup* _inputGroup;
        NeuralGroup* _outputGroup;

        map<string, NeuralGroup*> _groups;
        map<int, Connection*> _connections;
        map<int, Connection*> _recConnections;

        Vector _output;
    };
}

