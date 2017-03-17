//
// Created by mpechac on 10. 3. 2016.
//

#include <fstream>
#include <algorithm>
#include <assert.h>
#include "Dataset.h"
#include "StringUtils.h"

using namespace NeuroNet;

Dataset::Dataset() {
}

Dataset::~Dataset() {

}

void Dataset::load(string p_filename, DatasetConfig p_format) {
    string line;
    ifstream file(p_filename);
    _config = p_format;
    if (file.is_open())
    {
        while ( getline (file,line) )
        {
            if (line[0] != '@' && StringUtils::trim(line).length() > 0) {
                parseLine(line, _config.delimiter);
            }
        }
        file.close();
    }
}

void Dataset::parseLine(string p_line, string p_delim) {
    vector<string> tokens;
    vector<string> targets;
    int index = 0;
    size_t pos = 0;
    string token;

    if (p_delim == "") {
        tokens.push_back(StringUtils::trim(p_line));
    }
    else {
        while ((pos = p_line.find(p_delim)) != std::string::npos) {
            token = p_line.substr(0, pos);
            if (index >= _config.targetPos && index < _config.targetPos + _config.targetDim) {
                targets.push_back(StringUtils::trim(token));
            }
            else {
                tokens.push_back(StringUtils::trim(token));
            }
            p_line.erase(0, pos + p_delim.length());
        }
    }

    Vector sample(_config.inDim);
    Vector target(_config.targetDim);

    for(int i = 0; i < _config.inDim; i++) {
        sample[i] = stod(tokens[i]);
    }

    for(int i = 0; i < targets.size(); i++) {
        target[i] = stod(targets[i]);
    }

    _buffer.push_back(pair<Vector, Vector>(sample, target));
}

void Dataset::normalize() {
    Vector max = Vector::Zero(_config.inDim);

    for(int i = 0; i < _buffer.size(); i++) {
        for(int j = 0; j < _config.inDim; j++) {
            if (max[j] < _buffer[i].first[j]) {
                max[j] = _buffer[i].first[j];
            }
        }
    }

    for(int i = 0; i < _buffer.size(); i++) {
        for (int j = 0; j < _config.inDim; j++) {
            _buffer[i].first[j] /= max[j];
            if (_buffer[i].first[j] != _buffer[i].first[j]) {
                assert(0);
            }
        }
    }
}

void Dataset::permute() {
    random_shuffle(_buffer.begin(), _buffer.end());
}
