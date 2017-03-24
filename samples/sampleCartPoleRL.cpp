//
// Created by mpechac on 21. 3. 2017.
//


#include "sampleCartPoleRL.h"
#include "CartPoleTask.h"
#include "../network/NeuralNetwork.h"
#include "../algorithm/optimizer/BackProp.h"
#include "../algorithm/rl/TD.h"
#include "../algorithm/rl/CACLAActor.h"
#include "../log/log.h"
#include "../network/NetworkUtils.h"
#include "../backend/sflab/RandomGenerator.h"
#include "../algorithm/optimizer/ADAM.h"
#include "../algorithm/rl/QLearning.h"

sampleCartPoleRL::sampleCartPoleRL() {

}

sampleCartPoleRL::~sampleCartPoleRL() {

}

Vector sampleCartPoleRL::encodeState(vector<double> *p_sensors) {
    Vector res;
    Vector encoded(16);

    NetworkUtils::gaussianEncoding(p_sensors->at(0), -0.7, 0.7, 16, .5, &encoded);
    res = encoded;
    //cout << "Value 1: " << p_sensors->at(0) << endl;
    //cout << encoded << endl << endl;
    NetworkUtils::gaussianEncoding(p_sensors->at(1), -10, 10, 16, .5, &encoded);
    res = Vector::Concat(res, encoded);
    //cout << "Value 2: " << p_sensors->at(1) << endl;
    //cout << encoded << endl << endl;
    NetworkUtils::gaussianEncoding(p_sensors->at(2), -2.4, 2.4, 16, .5, &encoded);
    res = Vector::Concat(res, encoded);
    //cout << "Value 3: " << p_sensors->at(2) << endl;
    //cout << encoded << endl << endl;
    NetworkUtils::gaussianEncoding(p_sensors->at(3), -0.5, 0.5, 16, .5, &encoded);
    res = Vector::Concat(res, encoded);
    //cout << "Value 4: " << p_sensors->at(3) << endl;
    //cout << encoded << endl << endl;

    return Vector(res);
}

Vector sampleCartPoleRL::chooseAction(Vector &p_action, double epsilon) {

    Vector action(p_action.size());

    //cout << p_action << endl;

    for(int i = 0; i < p_action.size(); i++) {
        action[i] = RandomGenerator::getInstance().normalRandom(p_action[i], epsilon);
    }

    //cout << action << endl;

    return Vector(action);
}

void sampleCartPoleRL::sampleCACLA() {
    CartPoleTask task;
    CartPole *cartPole = task.getEnvironment();

    double gamma = 0.9;
    NeuralNetwork nc;

    nc.addLayer("input", 64, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    nc.addLayer("hidden0", 164, NeuralGroup::RELU, NeuralNetwork::HIDDEN);
    nc.addLayer("hidden1", 150, NeuralGroup::RELU, NeuralNetwork::HIDDEN);
    nc.addLayer("output", 1, NeuralGroup::LINEAR, NeuralNetwork::OUTPUT);

    // feed-forward connections
    nc.addConnection("input", "hidden0");
    nc.addConnection("hidden0", "hidden1");
    nc.addConnection("hidden1", "output");

    ADAM optimizer_c(&nc);
    //BackProp optimizer_c(&nc, 1e-6, 0.9, true);
    TD critic(&optimizer_c, &nc, gamma);
    critic.setAlpha(0.001);

    NeuralNetwork na;

    na.addLayer("input", 64, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    na.addLayer("hidden0", 164, NeuralGroup::RELU, NeuralNetwork::HIDDEN);
    na.addLayer("hidden1", 150, NeuralGroup::RELU, NeuralNetwork::HIDDEN);
    na.addLayer("output", 1, NeuralGroup::LINEAR, NeuralNetwork::OUTPUT);

    // feed-forward connections
    na.addConnection("input", "hidden0");
    na.addConnection("hidden0", "hidden1");
    na.addConnection("hidden1", "output");

    ADAM optimizer_a(&na);
    //BackProp optimizer_a(&nc, 1e-6, 0.9, true);
    CACLAActor actor(&optimizer_a, &na);
    actor.setAlpha(0.001);


    vector<double> sensors;
    Vector state0, state1;
    Vector action0(1);
    double reward = 0;
    double value0, value1;
    double delta;
    double epsilon = 1;
    int epochs = 1000;

    int wins = 0, loses = 0;

    FILE* pFile = fopen("application.log", "w");
    Output2FILE::Stream() = pFile;
    FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

    for (int e = 0; e < epochs; e++) {
        cout << "Epoch " << e << endl;

        task.reset();

        while(!task.isFinished()) {
            //cout << cartPole->toString() << endl;

            sensors = cartPole->getSensors();
            state0 = encodeState(&sensors);
            nc.activate(&state0);
            na.activate(&state0);
            action0 = chooseAction((*na.getOutput()), epsilon);
            value0 = (*nc.getOutput())[0];
            //cout << action0[0] << endl;
            cartPole->performAction(action0[0]);

            sensors = cartPole->getSensors();
            state1 = encodeState(&sensors);
            nc.activate(&state1);
            value1 = (*nc.getOutput())[0];
            reward = task.getReward();
            delta = reward + gamma * value1 - value0;

            critic.train(&state0, &state1, reward);
            actor.train(&state0, &action0, delta);

            //FILE_LOG(logDEBUG1) << action0[0];
        }

        if (!task.failed()) {
            wins++;
        }
        else {
            loses++;
        }

        cout << task.getT() << endl;
        cout << wins << " / " << loses << endl;
        FILE_LOG(logDEBUG1) << wins << " " << loses;


        if (epsilon > 0.01) {
            epsilon -= (1.0 / epochs);
        }
    }
}
