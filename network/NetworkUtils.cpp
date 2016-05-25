//
// Created by mpechac on 23. 3. 2016.
//

#include "NetworkUtils.h"
#include "json.hpp"
#include "Define.h"
#include <iostream>
#include <fstream>

using namespace NeuroNet;
using json = nlohmann::json;

int NetworkUtils::kroneckerDelta(int p_i, int p_j) {
 return p_i == p_j ? 1 : 0;
}

void NetworkUtils::coarseEncoding(double p_value, double p_upperLimit, double p_lowerLimit, double p_populationDim, VectorXd *p_vector) {

}

void NetworkUtils::binaryEncoding(double p_value, VectorXd *p_vector) {
 p_vector->fill(0);
 (*p_vector)[p_value] = 1;
}

void NetworkUtils::saveNetwork(string p_filename, NeuralNetwork *p_network) {

 if (p_filename.find(".net") == p_filename.size()) {
  p_filename += ".net";
 }

 json data;

 data["_header"] = "NeuroNet";
 data["_version"] = VERSION;
 data["_network"] = p_network->getFileData();
 for (auto it = p_network->getGroups()->begin(); it != p_network->getGroups()->end(); it++ ) {
  NeuralGroup* group = it->second;
  data["layers"][group->getId()] = group->getFileData();
 }
 for (auto it = p_network->getConnections()->begin(); it != p_network->getConnections()->end(); it++ ) {
  Connection* connection = it->second;
  data["connections"][to_string(connection->getId())] = connection->getFileData();
 }

 ofstream file;
 file.open (p_filename);
 file << data.dump();
 file.close();
}
