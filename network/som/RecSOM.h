//
// Created by mpechac on 19. 5. 2016.
//

#ifndef NEURONET_RECSOM_H
#define NEURONET_RECSOM_H

#include "SOM.h"

using json = nlohmann::json;

namespace NeuroNet {

class RecSOM : public SOM {
public:
    RecSOM(int p_dimInput, int p_dimX, int p_dimY, NeuralGroup::ACTIVATION p_actFunction);
    ~RecSOM();

    void train(Vector *p_input) override ;
    void activate(Vector *p_input) override;

    void initTraining(double p_gamma1, double p_gamma2, double p_alpha, double p_beta, double p_epochs);
    void initTraining(double p_alpha, double p_epochs) override;

    void paramDecay() override;

    json getFileData() override;
private:
    void updateWeights() override;
    void updateContext();
    double calcDistance(int p_index) override;

    double _beta;
    double _gamma1_0;
    double _gamma1;
    double _gamma2_0;
    double _gamma2;
};

}


#endif //NEURONET_RECSOM_H
