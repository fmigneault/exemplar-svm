/* -------------------------
Logging/printing operations
------------------------- */
#ifndef LOGGER_H
#define LOGGER_H

// required magic!
#define __STDC_WANT_LIB_EXT1__ 1

#include "esvmTypes.h"
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <sstream>

const std::string WINDOW_NAME = "Display";
const std::string LOGGER_FILE = "output.txt";

std::string currentTimeStamp();

class logstream
{
public:
    bool useConsoleOutput;
    bool useFileOutput;
    std::ofstream coss;
    logstream(std::string filepath, bool useConsoleOutput = true, bool useFileOutput = true);
    ~logstream(void);
    logstream& operator<< (std::ostream& (*pfun)(std::ostream&));
};

template <class T>
logstream& operator<< (logstream& log, const T val)
{    
    if (log.useFileOutput)      log.coss << val;
    if (log.useConsoleOutput)   std::cout << val;
    return log;
}

template <class T>
logstream& operator<< (logstream& log, const std::vector<T>& v)
{
    std::ostringstream oss;
    oss << "[";
    typename std::vector<T>::const_iterator it;
    for (it = v.begin(); it != v.end()-1; ++it)
        oss << *it << " ";
    oss << *it << "]";

    std::string s = oss.str();
    if (log.useFileOutput)      log.coss << s;
    if (log.useConsoleOutput)   std::cout << s;
    return log;
}

std::string featuresToVectorString(FeatureVector features);
std::string featuresToSvmString(FeatureVector features, int label);

#endif/*LOGGER_H*/
