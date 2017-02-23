#ifndef GENERIC_H
#define GENERIC_H

#include "logger.h"
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>

/* --------------------
Search/sort operations
---------------------*/

// Search container for value
template<class C, class T>
inline bool contains(const C& v, const T& x) { return end(v) != std::find(begin(v), end(v), x); }

class SequenceGen
{
public:
    SequenceGen(int start = 0) : current(start) {}
    int operator() () { return current++; }
private:
    int current;
};

template<typename T>
class Compare
{
public:    
    Compare(std::vector<T, std::allocator<T> >& v) : _v(v) {}
    inline bool operator()(size_t i, size_t j) const { return _v[i] < _v[j]; }
private:
    std::vector<T, std::allocator<T> >& _v;
};

/* --------------------
Compare
---------------------*/
inline bool doubleAlmostEquals(double x1, double x2)
{
    static double eps = std::numeric_limits<double>::epsilon();
    return std::abs(x1 - x2) < std::abs(std::min(x1, x2)) * eps;
}

/* --------------------
Asserts
---------------------*/ 

// Assert with message printing
#define ASSERT_THROW(cond, msg) do \
{ if (!(cond)) { \
    std::ostringstream oss; \
    oss << msg; \
    std::string str = "Assertion failed: " + oss.str(); \
    throw std::runtime_error(str); } \
} while(0)

// Assert with message printing
#define ASSERT_MSG(cond, msg) do \
{ if (!(cond)) { \
    std::ostringstream oss; \
    oss << msg; \
    std::string str = "Assertion failed: " + oss.str(); \
    std::cerr << str << std::endl; \
    throw std::runtime_error(str); } \
} while(0)

// Assert with message printing and logging
#define ASSERT_LOG(cond, msg) do \
{ if (!(cond)) { \
    logstream log(LOGGER_FILE); \
    std::ostringstream oss; \
    oss << msg; \
    std::string str = "Assertion failed: " + oss.str(); \
    log << str << std::endl; \
    throw std::runtime_error(str); } \
} while(0)

#endif/*GENERIC_H*/
