//
// Created by mpechac on 23. 3. 2016.
//

#include "NetworkUtils.h"
#include "json.hpp"
#include "Define.h"
#include "som/RecSOM.h"
#include "../dataset/StringUtils.h"
#include "som/MSOM.h"
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

NeuralNetwork* NetworkUtils::loadNetwork(string p_filename) {
 json data;
 ifstream file;

 if (p_filename.find(".net") == p_filename.size()) {
  p_filename += ".net";
 }

 file.open (p_filename);
 file >> data;
 file.close();

 string version = data["_version"];

 if (version.compare(VERSION) != 0) {
  cout << "Warning: versions could not be compatible" << endl;
 }

 string type = data["_network"]["type"];

 if (type.compare("feedforward") == 0) {

  NeuralNetwork* network = new NeuralNetwork();

  string inGroupId = data["_network"]["ingroup"];
  string outGroupId = data["_network"]["outgroup"];

  for (json::iterator it = data["layers"].begin(); it != data["layers"].end(); ++it) {
   string id = it.key();
   json group = it.value();

   if (id.compare(inGroupId) == 0) {
     network->addLayer(id, group["dim"], group["actfn"], NeuralNetwork::INPUT);
   }
   else if (id.compare(outGroupId) == 0) {
    network->addLayer(id, group["dim"], group["actfn"], NeuralNetwork::OUTPUT);
   }
   else {
    network->addLayer(id, group["dim"], group["actfn"], NeuralNetwork::HIDDEN);
   }
  }

  for (json::iterator it = data["connections"].begin(); it != data["connections"].end(); ++it) {
   json connection = it.value();

   MatrixXd* weights = new MatrixXd(network->getGroup(connection["outgroup"])->getDim(), network->getGroup(connection["ingroup"])->getDim());
   vector<string> weightsRaw = StringUtils::split(connection["weights"], '|');

   for(int i = 0; i < weights->rows(); i++) {
    for(int j = 0; j < weights->cols(); j++) {
     (*weights)(i,j) = stod(weightsRaw[i * weights->cols() + j]);
    }
   }

   network->addConnection(connection["ingroup"], connection["outgroup"])->init(weights);
  }

  return network;
 }

 if (type.compare("recsom") == 0) {
  int dimX = data["_network"]["dimx"].get<int>();
  int dimY = data["_network"]["dimy"].get<int>();

  json inputLayer = data["layers"].find("input").value();
  int dimInput = inputLayer["dim"].get<int>();
  json latticeLayer = data["layers"].find("lattice").value();
  int actFunction = latticeLayer["actfn"].get<int>();

  RecSOM* recSOM = new RecSOM(dimInput, dimX, dimY, actFunction);

  for (json::iterator it = data["connections"].begin(); it != data["connections"].end(); ++it) {
   json connection = it.value();

   MatrixXd* weights = new MatrixXd(recSOM->getGroup(connection["outgroup"])->getDim(), recSOM->getGroup(connection["ingroup"])->getDim());
   vector<string> weightsRaw = StringUtils::split(connection["weights"], '|');

   for(int i = 0; i < weights->rows(); i++) {
     for(int j = 0; j < weights->cols(); j++) {
       (*weights)(i,j) = stod(weightsRaw[i * weights->cols() + j]);
     }
   }

   recSOM->getConnection(connection["ingroup"], connection["outgroup"])->init(weights);
  }

  return recSOM;
 }

 if (type.compare("msom") == 0) {
  int dimX = data["_network"]["dimx"].get<int>();
  int dimY = data["_network"]["dimy"].get<int>();

  json inputLayer = data["layers"].find("input").value();
  int dimInput = inputLayer["dim"].get<int>();
  json latticeLayer = data["layers"].find("lattice").value();
  int actFunction = latticeLayer["actfn"].get<int>();

  MSOM* mSOM = new MSOM(dimInput, dimX, dimY, actFunction);

  for (json::iterator it = data["connections"].begin(); it != data["connections"].end(); ++it) {
   json connection = it.value();

   MatrixXd* weights = new MatrixXd(mSOM->getGroup(connection["outgroup"])->getDim(), mSOM->getGroup(connection["ingroup"])->getDim());
   vector<string> weightsRaw = StringUtils::split(connection["weights"], '|');

   for(int i = 0; i < weights->rows(); i++) {
    for(int j = 0; j < weights->cols(); j++) {
     (*weights)(i,j) = stod(weightsRaw[i * weights->cols() + j]);
    }
   }

   mSOM->getConnection(connection["ingroup"], connection["outgroup"])->init(weights);
  }

  return mSOM;
 }

 return nullptr;
}
