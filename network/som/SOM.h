//
// Created by mpechac on 8. 3. 2016.
//

#ifndef LIBNEURONET_SOM_H
#define LIBNEURONET_SOM_H


#include <set>
#include "../NeuralNetwork.h"

using namespace std;

namespace NeuroNet {

class SOM : public NeuralNetwork {
public:
    enum NEIGHBORHOOD_TYPE {
        EUCLIDEAN = 0,
        GAUSSIAN = 1
    };

    SOM(int p_dimInput, int p_dimX, int p_dimY, int p_actFunction);
    ~SOM(void);

    virtual void train(double *p_input);
    virtual void activate(VectorXd *p_input) override;
    virtual void initTraining(double p_alpha, double p_epochs);
    virtual void paramDecay();

    double getError() { return _qError; };
    double getWinnerDifferentiation();

protected:
    virtual void updateWeights();
    virtual void findWinner();
    virtual double calcDistance(int p_index);
    double calcNeighborhood(int p_index, NEIGHBORHOOD_TYPE p_type);
    double vectorDistance(VectorXd* p_v1, VectorXd* p_v2);
    double euclideanDistance(int p_x1, int p_y1, int p_x2, int p_y2);
    double gaussianDistance(int p_d, double p_sigma = 1);

protected:
    double _sigma;
    double _sigma0;
    double _lambda;
    double _alpha0;
    double _alpha;
    double _iteration;

    double _qError;
    set<int> _winnerSet;

    int _winner;
    int _dimX;
    int _dimY;
};

}

#endif //LIBNEURONET_SOM_H
