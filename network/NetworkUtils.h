//
// Created by mpechac on 23. 3. 2016.
//

#ifndef NEURONET_NETWORKUTILS_H
#define NEURONET_NETWORKUTILS_H

#include <Eigen/Dense>

using namespace Eigen;

class NetworkUtils {
public:
    NetworkUtils() {};
    ~NetworkUtils() {};

    static void coarseEncoding(double p_value, double p_upperLimit, double p_lowerLimit, double p_populationDim, VectorXd& p_vector);
    static int kroneckerDelta(int p_i, int p_j);

    template <typename T>
    static int sgn(T val) {
      return (T(0) < val) - (val < T(0));
    }
};


#endif //NEURONET_NETWORKUTILS_H
