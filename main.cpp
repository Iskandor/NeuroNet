#include <iostream>
#include <ctime>
#include "samples/samples.h"
#include "network/RandomGenerator.h"
#include "backend/sflab/Matrix.h"

using namespace std;
using namespace SFLAB;

int main(int argc, char* argv[])
{
    std::clock_t start;
    double duration;

    Vector vector1(3, Base::VALUE, 1);
    Vector vector2(3, Base::VALUE, 2);
    Matrix matrix1(3, 3, Base::VALUE, 2);

    start = std::clock();

    cout << matrix1.inv() << endl;

    duration = ( std::clock() - start );

    std::cout<<"printf: "<< duration <<'\n';


    //sampleBP();
    //sampleSOM();
    //sampleMSOM();
    //sampleQ();
    //sampleSARSA();
    //sampleAC();

    system("pause");
    return 0;
}