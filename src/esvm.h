#ifndef ESVM_LIBSVM_H
#define ESVM_LIBSVM_H

#include "esvmTypesDef.h"
#include "svm.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class ESVM
{
public:
    inline ESVM() {}
    ESVM(std::vector< FeatureVector > positives, std::vector< FeatureVector > negatives, std::string id);
    ESVM(std::string filename, std::string id);
    double predict(FeatureVector sample);
    std::vector<double> predict(std::string filename, std::vector<int>& classGroundTruths);
    inline std::string getTargetID() { return targetID; }

private:
    void trainEnsembleModel(std::vector< FeatureVector > samples, std::vector<int> outputs,
                            int positiveOutput, int negativeOutput, double positiveWeight, double negativeWeight);
    static svm_node* getFeatureVector(FeatureVector features);
    static svm_node* getFeatureVector(double* features, int featureCount);
    svm_model* ensembleModel = nullptr;
    std::string targetID;
};

#endif/*ESVM_LIBSVM_H*/
