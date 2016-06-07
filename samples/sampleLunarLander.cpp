//
// Created by mpechac on 23. 3. 2016.
//

#include "../network/NeuralNetwork.h"
#include "../network/Define.h"
#include "../algorithm/rl/QLearning.h"
#include "LunarLander.h"
#include "../log/log.h"
#include "../network/som/SOM.h"
#include "../network/filters/NormalizationFilter.h"
#include "../algorithm/GreedyPolicy.h"
#include "../algorithm/rl/CACLA.h"

void sampleLunarLander() {
    double sumReward = 0;
    int time = 0;
    int dim = 50+20+20;
    int episode = 0;
    const int EPISODES = 100000;
    const int SIZE = 9;

    NeuralNetwork critic;
    critic.addLayer("input", dim+2, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    critic.addLayer("biasH", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    critic.addLayer("biasO", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    critic.addLayer("hidden", 5, NeuralGroup::SIGMOID, NeuralNetwork::HIDDEN);
    critic.addLayer("output", 1, NeuralGroup::TANH, NeuralNetwork::OUTPUT);

    //VectorXd limit(dim+2);
    //limit << 20,50,20,1,1;
    //critic.getGroup("input")->addInFilter(new NormalizationFilter(&limit));
    // feed-forward connections
    critic.addConnection("input", "hidden");
    critic.addConnection("hidden", "output");
    // bias connections
    critic.addConnection("biasH", "hidden");
    critic.addConnection("biasO", "output");

    NeuralNetwork actor;
    actor.addLayer("input", dim, NeuralGroup::IDENTITY, NeuralNetwork::INPUT);
    actor.addLayer("biasH", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    actor.addLayer("biasO", 1, NeuralGroup::BIAS, NeuralNetwork::HIDDEN);
    actor.addLayer("hidden", 4, NeuralGroup::SIGMOID, NeuralNetwork::HIDDEN);
    actor.addLayer("output", 2, NeuralGroup::SIGMOID, NeuralNetwork::OUTPUT);
    // feed-forward connections
    actor.addConnection("input", "hidden");
    actor.addConnection("hidden", "output");
    // bias connections
    actor.addConnection("biasH", "hidden");
    actor.addConnection("biasO", "output");

    CACLA agent(&actor, &critic);
    LunarLander lander;

    agent.setAlpha(0.5);
    agent.setBeta(0.1);
    agent.setExploration(0.01);
    agent.init(&lander);

    FILE* pFile = fopen("application.log", "w");
    Output2FILE::Stream() = pFile;
    FILELog::ReportingLevel() = FILELog::FromString("DEBUG1");

    while(episode < 100000) {

        agent.run();
        sumReward += lander.getReward();

        // 3. update
        time++;

        // 4. check whether terminal state was reached
        if (lander.isFinished()) {
            cout << "Finished episode " << episode << "! " << time << " Reward:" << sumReward << endl;
            FILE_LOG(logDEBUG1) << sumReward;

            time = 0;
            sumReward = 0;
            lander.reset();
            episode++;
        }
    }

    cout << "Uspesne ukoncene." << endl;
}
