//
// Created by mpechac on 23. 3. 2016.
//

#include "NetworkUtils.h"

using namespace NeuroNet;

int NetworkUtils::kroneckerDelta(int p_i, int p_j) {
 return p_i == p_j ? 1 : 0;
}

void NetworkUtils::coarseEncoding(double p_value, double p_upperLimit, double p_lowerLimit, double p_populationDim, VectorXd *p_vector) {

}

void NetworkUtils::binaryEncoding(double p_value, VectorXd *p_vector) {
 p_vector->fill(0);
 (*p_vector)[p_value] = 1;
}
