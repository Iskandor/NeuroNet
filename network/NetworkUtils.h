//
// Created by mpechac on 23. 3. 2016.
//

#ifndef NEURONET_NETWORKUTILS_H
#define NEURONET_NETWORKUTILS_H

#include <Eigen/Dense>
#include "NeuralNetwork.h"

using namespace Eigen;
using namespace std;

namespace NeuroNet {

class NetworkUtils {
public:
    NetworkUtils() {};
    ~NetworkUtils() {};

    static void saveNetwork(string p_filename, NeuralNetwork *p_network);
    static NeuralNetwork* loadNetwork(string p_filename);

    static void binaryEncoding(double p_value, VectorXd* p_vector);
    static void gaussianEncoding(double p_value, double p_upperLimit, double p_lowerLimit, int p_populationDim, VectorXd* p_vector);
    static int kroneckerDelta(int p_i, int p_j);

    template <typename T>
    static int sgn(T val) {
      return (T(0) < val) - (val < T(0));
    }
};

}

#endif //NEURONET_NETWORKUTILS_H
