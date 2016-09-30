//
// Created by mpechac on 29. 6. 2016.
//

#include "samples.h"
#include "../network/NeuralNetwork.h"
#include "Maze.h"
#include "../log/log.h"
#include "../algorithm/rl/RegularGradientActor.h"
#include "../network/NetworkUtils.h"
#include "../network/RandomGenerator.h"
#include "../network/filters/KwtaFilter.h"
#include "../algorithm/rl/NaturalGradientActor.h"

using namespace NeuroNet;

void sampleActor() {

    double sumReward = 0;
    int time = 0;
    int dim = 3;

    NeuralNetwork* actor = new NeuralNetwork();
    actor->addLayer("input", dim*dim, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    actor->addLayer("biasH", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    actor->addLayer("biasO", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    actor->addLayer("hidden", 90, NeuralGroup::TANH, NeuralNetwork::HIDDEN);
    actor->addLayer("output", 4, NeuralGroup::SIGMOID, NeuralNetwork::OUTPUT);
    actor->addConnection("input", "hidden");
    actor->addConnection("hidden", "output");
    actor->addConnection("biasH", "hidden");
    actor->addConnection("biasO", "output");
    //actor->getGroup("output")->addOutFilter(new KwtaFilter(1, true));

    Maze maze(dim);
    maze.reset();

    FILE* pFile = fopen("application.log", "w");
    Output2FILE::Stream() = pFile;
    FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

    int episode = 0;
    double epsilon = 0.01;
    double reward = 0;
    double tdError = 0;

    VectorXd action(4);
    VectorXd state0(dim*dim);
    double stateV0;
    double stateV1;

    RegularGradientActor RGA(actor);
    RGA.setAlpha(.1); //.000001

    while(episode < 2000) {

        state0 = *maze.getState();
        stateV0 = maze.getStateValue();
        actor->activate(&state0);
        int action_i = 0;

        /*
        for(auto i = 0; i < dim; i++) {
            for(auto j = 0; j < dim; j++) {
                cout << (*maze.getState())(i*dim + j);
            }
            cout << endl;
        }
        cout << endl << "output" << endl;
        cout << *actor->getGroup("output")->getOutput() << endl;
        system("pause");
        */

        for (int i = 0; i < action.size(); i++) {
            if ((*actor->getOutput())[action_i] < (*actor->getOutput())[i]) {
                action_i = i;
            }
        }

        if (RandomGenerator::getInstance().random() < epsilon) {
            action_i = RandomGenerator::getInstance().random(0, 3);
        }
        action.fill(0);
        action[action_i] = 1;

        maze.updateState(&action);
        stateV1 = maze.getStateValue();
        reward = maze.getReward();
        sumReward += reward;
        tdError = reward + 0.99 * stateV1 - stateV0;

        // 3. update
        RGA.train(&state0, tdError);
        time++;

        // 4. check whether terminal state was reached
        if (maze.isFinished()) {

            cout << "Finished episode " << episode << "! " << time << " Reward:" << sumReward << endl;
            FILE_LOG(logDEBUG1) << sumReward;

            maze.reset();
            time = 0;
            sumReward = 0;
            episode++;
        }
    }

    NetworkUtils::saveNetwork("rga_actor.net", actor);

    delete actor;

    cout << "Uspesne ukoncene." << endl;


}
