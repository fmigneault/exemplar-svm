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
    ESVM(std::string filename, std::string id);
    double predict(FeatureVector sample);
    std::vector< std::tuple<double, int> > predict(std::string filename);
    inline std::string getTargetID() { return targetID; }

private:
    void trainEnsembleModel(std::vector< FeatureVector > samples, std::vector<int> outputs,
                            int positiveOutput, int negativeOutput, double positiveWeight, double negativeWeight);
    svm_node* getFeatureVector(FeatureVector features);
    svm_node* getFeatureVector(double* features, int featureCount);
    svm_model* ensembleModel = nullptr;
    std::string targetID;
};

#endif/*ESVM_LIBLINEAR_H*/
