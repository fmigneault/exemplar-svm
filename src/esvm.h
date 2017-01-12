#ifndef ESVM_LIBLINEAR_H
#define ESVM_LIBLINEAR_H

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "svm.h"
#include "esvmTypesDef.h"

class ESVM
{
public:
    inline ESVM() {}
    ESVM(std::vector< FeatureVector > positives, std::vector< FeatureVector > negatives, std::string id);
    double predict(FeatureVector sample);
    inline std::string getTargetID() { return targetID; }

private:    
    void trainEnsembleModel(std::vector< FeatureVector > samples, std::vector<int> outputs, double positiveWeight, double negativeWeight);
    svm_node* getFeatureVector(FeatureVector features);
    svm_node* getFeatureVector(double* features, int featureCount);
    svm_model* ensembleModel;
    std::string targetID;
};

#endif/*ESVM_LIBLINEAR_H*/
