#include <iostream>
#include <ctime>
#include <c++/iomanip>
#include "samples/samples.h"
#include "backend/sflab/Matrix.h"
#include "environments/cartpole/rk4.hpp"

using namespace std;
using namespace SFLAB;


int main(int argc, char* argv[])
{
    /*
    std::clock_t start;
    double duration;

    SFLAB::Matrix matrix1(2, 2);
    SFLAB::Matrix matrix2(2, 2);

    matrix1[0][0] = 1;
    matrix1[0][1] = 2;
    matrix1[1][0] = 3;
    matrix1[1][1] = 4;

    matrix2[0][0] = 5;
    matrix2[0][1] = 6;
    matrix2[1][0] = 7;
    matrix2[1][1] = 8;

    MatrixXd emat1(2,2);
    MatrixXd emat2(2,2);

    emat1 << 1,2,3,4;
    emat2 << 5,6,7,8;

    start = std::clock();

    cout << matrix1.ew_dot(matrix2) << endl;

    cout << emat1.cwiseProduct(emat2) << endl;

    duration = ( std::clock() - start );

    std::cout<<"printf: "<< duration <<'\n';
    */

    //sampleBP();
    //sampleSOM();
    //sampleMSOM();
    //sampleQ();
    //sampleSARSA();
    //sampleAC();
    //sampleALE();
    sampleContinuousRL();

    system("pause");
    return 0;
}