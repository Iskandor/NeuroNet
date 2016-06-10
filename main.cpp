#include <iostream>
#include "samples/samples.h"
#include "network/RandomGenerator.h"

using namespace std;

int main(int argc, char* argv[])
{
    /*
    NeuroNet::RandomGenerator generator;
    for(int i = 0; i < 100; i++) {
        cout << generator.random() << endl;
    }
    */
    //sampleBP();
    //sampleTD();
    //sampleQ();
    //sampleSOM();
    //sampleMSOM();
    //sampleDigits();
    //sampleSARSA();
    //sampleTDAC();

    //sampleLunarLander();
    samplePoleCart();

    system("pause");
    return 0;
}