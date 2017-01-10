#ifndef ESVM_LIBLINEAR_H
#define ESVM_LIBLINEAR_H

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "svm.h"

class ESVM
{
public:
    ESVM(std::vector< std::vector<double> > positives, std::vector< std::vector<double> > negatives);
    double predict(std::vector<double> sample);

private:
    void trainEnsembleModel(std::vector< std::vector<double> > samples, std::vector<double> outputs);
    svm_node* getFeatureVector(std::vector<double> features);
    svm_node* getFeatureVector(double* features, int featureCount);
    svm_model* ensembleModel;
};

#endif/*ESVM_LIBLINEAR_H*/
