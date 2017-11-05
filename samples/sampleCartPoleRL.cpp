//
// Created by mpechac on 21. 3. 2017.
//


#include "sampleCartPoleRL.h"
#include "CartPoleTask.h"
#include "../network/NeuralNetwork.h"
#include "../algorithm/optimizer/BackProp.h"
#include "../algorithm/rl/TD.h"
#include "CACLA.h"
#include "../log/log.h"
#include "../network/NetworkUtils.h"
#include "../backend/FLAB/RandomGenerator.h"
#include "../algorithm/optimizer/ADAM.h"
#include "../algorithm/rl/QLearning.h"

sampleCartPoleRL::sampleCartPoleRL() {

}

sampleCartPoleRL::~sampleCartPoleRL() {

}

Vector sampleCartPoleRL::encodeState(vector<double> *p_sensors) {
    Vector res(4);
    /*
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
    //
     */

    res[0] = p_sensors->at(0) / 0.7;
    res[1] = p_sensors->at(1) / 50;
    res[2] = p_sensors->at(2) / 2.4;
    res[3] = p_sensors->at(3) / 50;

    /*
    cout << "Value 1: " << p_sensors->at(0) << endl;
    cout << "Value 2: " << p_sensors->at(1) << endl;
    cout << "Value 3: " << p_sensors->at(2) << endl;
    cout << "Value 4: " << p_sensors->at(3) << endl;
    */

    cout << res << endl;

    return Vector(res);
}

Vector sampleCartPoleRL::chooseAction(Vector &p_action, double epsilon) {

    Vector action(p_action.size());

    //cout << p_action << endl;

    for(int i = 0; i < p_action.size(); i++) {
        action[i] = RandomGenerator::getInstance().normalRandom(p_action[i], epsilon);
        //action[i] = p_action[i] + RandomGenerator::getInstance().random(-1.0, 1.0) * epsilon;
    }

    //cout << action << endl;

    return Vector(action);
}

void sampleCartPoleRL::sampleCACLA() {
    CartPoleTask task;
    CartPole *cartPole = task.getEnvironment();

    double gamma = 0.99;
    NeuralNetwork nc;

    nc.addLayer("input", 4, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    nc.addLayer("hidden0", 80, NeuralGroup::RELU, NeuralNetwork::HIDDEN);
    nc.addLayer("output", 1, NeuralGroup::TANH, NeuralNetwork::OUTPUT);

    // feed-forward connections
    nc.addConnection("input", "hidden0");
    nc.addConnection("hidden0", "output");

    ADAM optimizer_c(&nc);
    //BackProp optimizer_c(&nc, 1e-6, 0.9, true);
    TD critic(&optimizer_c, &nc, gamma);
    critic.init(0.001);

    NeuralNetwork na;

    na.addLayer("input", 4, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    na.addLayer("hidden0", 80, NeuralGroup::RELU, NeuralNetwork::HIDDEN);
    na.addLayer("output", 1, NeuralGroup::TANH, NeuralNetwork::OUTPUT);

    // feed-forward connections
    na.addConnection("input", "hidden0");
    na.addConnection("hidden0", "output");

    ADAM optimizer_a(&na);
    //BackProp optimizer_a(&na, 1e-6, 0.9, true);
    CACLA actor(&optimizer_a, &na);
    actor.init(0.001);


    vector<double> sensors;
    Vector state0, state1;
    Vector action0(1);
    double reward = 0;
    double value0, value1;
    double delta;
    double epsilon = 1;
    int epochs = 10000;

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
        FILE_LOG(logDEBUG1) << task.getT();


        if (epsilon > 0.01) {
            epsilon -= (1.0 / epochs);
        }
    }
}
