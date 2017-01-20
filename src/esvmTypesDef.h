#ifndef EXEMPLAR_SVM_TYPES_DEFINITIONS_H
#define EXEMPLAR_SVM_TYPES_DEFINITIONS_H

// OpenCV
#include "opencv2/opencv.hpp"

typedef std::vector<double> FeatureVector;

// ESVM options
#define ESVM_USE_HOG 1
#define ESVM_USE_LBP 0
#define ESVM_USE_PREDICT_PROBABILITY 0
#define ESVM_READ_DATA_FILES 0
#define ESVM_WRITE_DATA_FILES 0
#define ESVM_POSITIVE_CLASS +1
#define ESVM_NEGATIVE_CLASS -1
#define ESVM_WEIGHTS_MODE 0     // 0: (Wp = 0, Wn = 0), 1: (Wp = 1, Wn = 0.01), 2: (Wp = N/Np, Wn = N/Nn), 3: (Wp = 1, Wn = Np/Nn)

#endif/*EXEMPLAR_SVM_TYPES_DEFINITIONS_H*/