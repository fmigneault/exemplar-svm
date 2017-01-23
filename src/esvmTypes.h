#ifndef ESVM_TYPES_H
#define ESVM_TYPES_H

#include <vector>
#include "opencv2/opencv.hpp"

// Multi-Level vectors
template<typename T>
typedef std::vector<T>            Vector1;
template<typename T>
typedef std::vector< Vector1<T> > Vector2;
template<typename T>
typedef std::vector< Vector2<T> > Vector3;
template<typename T>
typedef std::vector< Vector3<T> > Vector4;

typedef Vector1<double> FeatureVector;

class ESVM;                             // Forward declaration
typedef Vector3<ESVM> EnsembleESVM;     // ESVM models [target][patch][feature]

#endif/*ESVM_TYPES_H*/
