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
    std::ofstream coss;
    logstream(std::string filepath, bool useConsoleOutput = true, bool useFileOutput = true);
    ~logstream(void);
    logstream& operator<< (std::ostream& (*pfun)(std::ostream&));
private:
    bool consoleOutput;
    bool fileOutput;
};

template <class T>
inline logstream& operator<< (logstream& log, T val)
{
    
    if (log.fileOutput)     log.coss << val;
    if (log.consoleOutput)  std::cout << val;
    return log;
}

template <class T>
inline logstream& operator<< (logstream& log, const std::vector<T>& v)
{
    std::ostringstream oss;
    oss << "[";
    typename std::vector<T>::const_iterator it;
    for (it = v.begin(); it != v.end()-1; ++it)
        oss << *it << " ";
    oss << *it << "]";

    std::string s = oss.str();
    if (log.fileOutput)     log.coss << s;
    if (log.consoleOutput)  std::cout << s;
    return log;
}

std::string featuresToVectorString(FeatureVector features);
std::string featuresToSvmString(FeatureVector features, int label);

#endif/*LOGGER_H*/
