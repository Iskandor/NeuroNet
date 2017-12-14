//
// Created by mpechac on 23. 3. 2016.
//

#ifndef NEURONET_NETWORKUTILS_H
#define NEURONET_NETWORKUTILS_H

#include "NeuralNetwork.h"

using namespace std;

namespace NeuroNet {

class NetworkUtils {
public:
    NetworkUtils() {};
    ~NetworkUtils() {};

    static void saveNetwork(string p_filename, NeuralNetwork *p_network);
    static NeuralNetwork* loadNetwork(string p_filename);

    static void binaryEncoding(double p_value, Vector* p_vector);
    static void gaussianEncoding(double p_value, double p_lowerLimit, double p_upperLimit, int p_populationDim, double p_variance, Vector* p_vector);
    static int kroneckerDelta(int p_i, int p_j);

    template <typename T>
    static int sgn(T val) {
      return (T(0) < val) - (val < T(0));
    }

    static time_t timestamp();
};

}

#endif //NEURONET_NETWORKUTILS_H
