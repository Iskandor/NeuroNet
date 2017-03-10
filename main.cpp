#include <iostream>
#include "samples/samples.h"
#include "network/RandomGenerator.h"

using namespace std;

int main(int argc, char* argv[])
{
    sampleBP();
    //sampleSOM();
    //sampleMSOM();
    sampleQ2();

    system("pause");
    return 0;
}