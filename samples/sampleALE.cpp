//
// Created by mpechac on 20. 3. 2017.
//
/*
#include <ale_interface.hpp>


using namespace std;

void sampleALE() {
    ALEInterface ale;

    // Get & Set the desired settings
    ale.setInt("random_seed", 123);
    //The default is already 0.25, this is just an example
    ale.setFloat("repeat_action_probability", 0.25);

#ifdef __USE_SDL
    ale.setBool("display_screen", true);
    ale.setBool("sound", true);
#endif

    // Load the ROM file. (Also resets the system for new settings to
    // take effect.)
    ale.loadROM("ALIEN.BIN");

    // Get the vector of legal actions
    ActionVect legal_actions = ale.getLegalActionSet();

    // Play 10 episodes
    for (int episode = 0; episode < 10; episode++) {
        float totalReward = 0;
        while (!ale.game_over()) {
            Action a = legal_actions[rand() % legal_actions.size()];
            // Apply the action and get the resulting reward
            float reward = ale.act(a);
            totalReward += reward;
        }
        cout << "Episode " << episode << " ended with score: " << totalReward << endl;
        ale.reset_game();
    }
}
 */