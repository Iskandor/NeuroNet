//
// Created by mpechac on 10. 3. 2016.
//

#include "StringUtils.h"

#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>

// trim from start
string& StringUtils::ltrim(string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// trim from end
string& StringUtils::rtrim(string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

// trim from both ends
string& StringUtils::trim(string &s) {
    return ltrim(rtrim(s));
}
