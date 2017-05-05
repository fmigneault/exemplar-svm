#ifndef ESVM_UTILS_H
#define ESVM_UTILS_H

#include "esvmTypes.h"
#include "esvmOptions.h"

#include "opencv2/objdetect.hpp"

#include "svm.h"
#include <string>

/* generic utilities / repetitive procedures */

cv::Mat esvmPreprocessFromMode(cv::Mat roi, cv::CascadeClassifier ccLocalSearch);

/* libsvm extra utilities */

std::string svm_type_name(svm_model*);
std::string svm_type_name(int /*svm_type*/);
std::string svm_kernel_name(svm_model*);
std::string svm_kernel_name(int /*kernel_type*/);

#endif/*ESVM_UTILS_H*/