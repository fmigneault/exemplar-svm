/* -------------------------
Logging/printing operations
------------------------- */
#ifndef LOGGER_H
#define LOGGER_H

#include "esvmTypes.h"
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>

const std::string WINDOW_NAME = "Display";
const std::string LOGGER_FILE = "output.txt";

std::string currentTimeStamp();

class logstream
{
public:
    std::ofstream coss;
    logstream(std::string filepath);
    ~logstream(void);
    logstream& operator<< (std::ostream& (*pfun)(std::ostream&));
};

template <class T>
logstream& operator<< (logstream& st, T val)
{
    st.coss << val;
    std::cout << val;
    return st;
}

std::string featuresToVectorString(FeatureVector features);
std::string featuresToSvmString(FeatureVector features, int label);

#endif/*LOGGER_H*/
