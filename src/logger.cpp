/* -------------------------
Logging/printing operations
------------------------- */
#include "logger.h"

std::string currentTimeStamp()
{
    time_t now = time(0);
    char dt[64];
    ctime_s(dt, sizeof dt, &now);
    return std::string(dt);
}

logstream::logstream(std::string filepath)
{
    // Open in append mode to log continously from different functions open/close calls
    coss.open(filepath, std::fstream::app);
}

logstream::~logstream(void)
{
    if (coss.is_open()) coss.close();
}

logstream& logstream::operator<<(std::ostream& (*pfun)(std::ostream&))
{
    pfun(coss);
    pfun(std::cout);
    return *this;
}

std::string featuresToVectorString(FeatureVector features)
{
    std::string s = "[" + std::to_string(features.size()) + "] {";
    for (int f = 0; f < features.size(); f++)
    {
        if (f != 0) s += ", ";
        s += std::to_string(features[f]);
    }
    s += "}";
    return s;
}

std::string featuresToSvmString(FeatureVector features, int label)
{
    std::string s = std::to_string(label) + " ";
    for (int f = 0; f < features.size(); f++)
    {
        if (f != 0) s += " ";
        s += std::to_string(f + 1) + ":";
        s += std::to_string(features[f]);
    }
    return s;
}