#include <iostream>
#include <ctime>
#include <c++/iomanip>
#include <samples/sampleGameRL.h>
#include <chrono>
#include <backend/base64.h>
#include <backend/flab/RandomGenerator.h>
#include <mns/ModelMNS.h>
#include "samples/samples.h"
#include "environments/cartpole/rk4.hpp"
#include "samples/sampleMazeRL.h"
#include "samples/sampleCartPoleRL.h"

using namespace std;
using namespace FLAB;
using namespace std::chrono;
using namespace MNS;

int main(int argc, char* argv[])
{
    /*
    for(int i = 0; i < 100; i++) {
        microseconds ms = duration_cast< microseconds >(system_clock::now().time_since_epoch());   // get time now
        string out;

        Base64::Encode(to_string(ms.count()), &out);

        cout << ms.count() << endl;
        cout << out << endl;
    }
    */

    ModelMNS model;

    model.init();
    model.run(1000);
    model.test();


    //sampleMazeRL sample;
    //sampleCartPoleRL sample;
    //sampleGameRL sample;

    //sampleBP();
    //sampleSOM();
    //sampleMSOM();
    //sample.sampleQ();
    //sample.sampleSARSA();
    //sample.sampleAC();
    //sample.sampleTD();
    //sampleALE();
    //sample.sampleCACLA();
    //sample.sampleTicTacToe();
    //sampleRTRL();

    system("pause");
    return 0;
}