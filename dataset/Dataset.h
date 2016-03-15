//
// Created by mpechac on 10. 3. 2016.
//

#ifndef LIBNEURONET_DATASET_H
#define LIBNEURONET_DATASET_H

#include <Eigen/Dense>
#include <vector>
#include "DatasetConfig.h"

using namespace std;
using namespace Eigen;

class Dataset {
public:
    Dataset();
    ~Dataset();

    void load(string p_filename, DatasetConfig p_format);
    void normalize();

    vector<pair<VectorXd, VectorXd>>* getData() { return &_buffer; };
    void permute();
protected:
    virtual void parseLine(string p_line, string p_delim);
private:
    DatasetConfig _config;
    vector<pair<VectorXd, VectorXd>> _buffer;
};


#endif //LIBNEURONET_DATASET_H
