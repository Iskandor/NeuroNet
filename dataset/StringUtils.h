//
// Created by mpechac on 10. 3. 2016.
//

#ifndef LIBNEURONET_STRINGUTILS_H
#define LIBNEURONET_STRINGUTILS_H

#include <string>

using namespace std;

class StringUtils {
public:
    StringUtils() {};
    ~StringUtils() {};
    static string &ltrim(string &s);
    static string &rtrim(string &s);
    static string &trim(string &s);
};


#endif //LIBNEURONET_STRINGUTILS_H
