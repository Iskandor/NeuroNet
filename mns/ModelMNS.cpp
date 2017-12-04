//
// Created by user on 5. 11. 2017.
//

#include <fstream>
#include <chrono>
#include "ModelMNS.h"

using namespace MNS;

ModelMNS::ModelMNS() {
    _msomMotor = nullptr;
    _msomVisual = nullptr;
}

ModelMNS::~ModelMNS() {
    delete _msomMotor;
    delete _msomVisual;
}

void ModelMNS::init() {
    _data.loadData("../data/Trajectories.3.vd", "../data/Trajectories.3.md");

    _msomMotor = new MSOM(16, _sizePMC, _sizePMC, NeuralGroup::EXPONENTIAL);
    _msomVisual = new MSOM(40, _sizeSTSp, _sizeSTSp, NeuralGroup::EXPONENTIAL);

}

void ModelMNS::run(int p_epochs) {
    _msomMotor->initTraining(0.01, 0.01, 0.3, 0.5, p_epochs);
    _msomVisual->initTraining(0.1, 0.1, 0.3 ,0.7, p_epochs);

    vector<Sequence*>* trainData = nullptr;

    for(int t = 0; t < p_epochs; t++) {
        cout << "Epoch " << t << endl;
        trainData = _data.permute();

        auto start = chrono::system_clock::now();
        for(int i = 0; i < trainData->size(); i++) {
            for(int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
                _msomMotor->train(trainData->at(i)->getMotorData()->at(j));
            }
            _msomMotor->resetContext();
            for(int p = 0; p < PERSPS; p++) {
                for(int j = 0; j < trainData->at(i)->getVisualData(p)->size(); j++) {
                    _msomVisual->train(trainData->at(i)->getVisualData(p)->at(j));
                }
                _msomVisual->resetContext();
            }
        }
        auto end = chrono::system_clock::now();
        chrono::duration<double> elapsed_seconds = end-start;
        cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
        cout << " PMC qError: " << _msomMotor->getError() << " WD: " << _msomMotor->getWinnerDifferentiation() << endl;
        cout << "STSp qError: " << _msomVisual->getError() << " WD: " << _msomVisual->getWinnerDifferentiation() << endl;
        _msomMotor->paramDecay();
        _msomVisual->paramDecay();
    }
}


void ModelMNS::test() {
    vector<Sequence*>* trainData = _data.permute();
    int winRatePMC_Motor[_sizePMC * _sizePMC][GRASPS];
    int winRateSTSp_Visual[_sizeSTSp * _sizeSTSp][PERSPS];
    int winRateSTSp_Motor[_sizeSTSp * _sizeSTSp][GRASPS];

    for(int i = 0; i < _sizePMC * _sizePMC; i++) {
        for(int j = 0; j < 3; j++) {
            winRatePMC_Motor[i][j] = 0;
        }
    }

    for(int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
        for(int j = 0; j < PERSPS; j++) {
            winRateSTSp_Visual[i][j] = 0;
        }
        for(int j = 0; j < GRASPS; j++) {
            winRateSTSp_Motor[i][j] = 0;
        }
    }

    for(int i = 0; i < trainData->size(); i++) {
        for(int j = 0; j < trainData->at(i)->getMotorData()->size(); j++) {
            _msomMotor->activate(trainData->at(i)->getMotorData()->at(j));
            winRatePMC_Motor[_msomMotor->getWinner()][trainData->at(i)->getGrasp() - 1]++;
        }
        _msomMotor->resetContext();
        for(int p = 0; p < PERSPS; p++) {
            for(int j = 0; j < trainData->at(i)->getVisualData(p)->size(); j++) {
                _msomVisual->activate(trainData->at(i)->getVisualData(p)->at(j));
                winRateSTSp_Visual[_msomVisual->getWinner()][p]++;
                winRateSTSp_Motor[_msomVisual->getWinner()][trainData->at(i)->getGrasp() - 1]++;
            }
        }
        _msomVisual->resetContext();
    }

    ofstream motFile("pmc.mot");

    if (motFile.is_open()) {
        motFile << _sizePMC << "," << _sizePMC << endl;
        for (int i = 0; i < _sizePMC * _sizePMC; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    motFile << winRatePMC_Motor[i][j];
                }
                else {
                    motFile << winRatePMC_Motor[i][j] << ",";
                }
            }
            if (i < _sizePMC * _sizePMC - 1) motFile << endl;
        }
    }

    motFile.close();

    ofstream STSvisFile("stsp.vis");

    if (STSvisFile.is_open()) {
        STSvisFile << _sizeSTSp << "," << _sizeSTSp << endl;
        for (int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
            for (int j = 0; j < PERSPS; j++) {
                if (j == PERSPS - 1) {
                    STSvisFile << winRateSTSp_Visual[i][j];
                }
                else {
                    STSvisFile << winRateSTSp_Visual[i][j] << ",";
                }
            }
            STSvisFile << endl;
        }
    }

    STSvisFile.close();

    ofstream STSmotFile("stsp.mot");

    if (STSmotFile.is_open()) {
        STSmotFile << _sizeSTSp << "," << _sizeSTSp << endl;
        for (int i = 0; i < _sizeSTSp * _sizeSTSp; i++) {
            for (int j = 0; j < GRASPS; j++) {
                if (j == GRASPS - 1) {
                    STSmotFile << winRateSTSp_Motor[i][j];
                }
                else {
                    STSmotFile << winRateSTSp_Motor[i][j] << ",";
                }
            }
            STSmotFile << endl;
        }
    }

    STSmotFile.close();
}
