#ifndef EXEMPLAR_SVM_TYPES_DEFINITIONS_H
#define EXEMPLAR_SVM_TYPES_DEFINITIONS_H

// OpenCV
#include "opencv2/opencv.hpp"

typedef std::vector<double> FeatureVector;

// ESVM options
#define USE_HOG 1
#define USE_LBP 0
#define SVM_WEIGHTS_MODE 0      // 0: unused, 1: (Cp = 1, Cn = 0.01), 2: (Cp = N/Np, Cn = N/Nn)
#define READ_DATA_FILES 0
#define WRITE_DATA_FILES 0

#endif/*EXEMPLAR_SVM_TYPES_DEFINITIONS_H*/