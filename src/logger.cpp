/* -------------------------
Logging/printing operations
------------------------- */
#include "logger.h"

std::string currentTimeStamp()
{
    time_t now = time(0);
    struct tm * timeinfo;
    timeinfo = localtime (&now);
    // char dt[64];
    // ctime_s(dt, sizeof dt, &now);
    char buffer[256];
    std::strftime(buffer, sizeof(buffer), "%a %b %d %H:%M:%S %Y", timeinfo);
    return std::string(buffer);
}

logstream::logstream(std::string filepath, bool useConsoleOutput, bool useFileOutput)
{
    fileOutput = useFileOutput;
    consoleOutput = useConsoleOutput;

    // Open in append mode to log continously from different functions open/close calls
    if (fileOutput)
        coss.open(filepath, std::fstream::app);
}

logstream::~logstream(void)
{
    if (coss.is_open()) coss.close();
}

logstream& logstream::operator<<(std::ostream& (*pfun)(std::ostream&))
{
    if (fileOutput)     pfun(coss);
    if (consoleOutput)  pfun(std::cout);
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