//
// Created by mpechac on 23. 3. 2016.
//

#include "NetworkUtils.h"

using namespace NeuroNet;

/*
void NetworkUtils::coarseEncoding(double p_value, double p_upperLimit, double p_lowerLimit, double p_populationDim, VectorXd &p_vector) {

}
 */

int NetworkUtils::kroneckerDelta(int p_i, int p_j) {
 return p_i == p_j ? 1 : 0;
}
