//
// Created by mpechac on 13. 4. 2016.
//

#ifndef NEURONET_MSOM_H
#define NEURONET_MSOM_H


#include "SOM.h"

using json = nlohmann::json;

namespace NeuroNet {

class MSOM : public SOM {
public:
    MSOM(int p_dimInput, int p_dimX, int p_dimY, NeuralGroup::ACTIVATION p_actFunction);
    ~MSOM();

    void train(Vector *p_input) override ;
    void activate(Vector *p_input) override;

    void initTraining(double p_gamma1, double p_gamma2, double p_alpha, double p_beta, double p_epochs);
    void initTraining(double p_alpha, double p_epochs) override;

    void paramDecay() override;
    void resetContext() override;

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
#endif //NEURONET_MSOM_H
