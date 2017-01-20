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
    ESVM(std::vector< FeatureVector > positives, std::vector< FeatureVector > negatives, std::string id = "");
    ESVM(std::string trainingFilePath, std::string id = "");
    ESVM(svm_model* trainedModel, std::string id = "");
    bool saveModelFile(std::string modelFilePath);
    double predict(FeatureVector sample);
    std::vector<double> predict(std::string filename, std::vector<int>& classGroundTruths);
    inline std::string getTargetID() { return targetID; }

private:
    void readSampleDataFile(std::string filePath, std::vector< FeatureVector >& sampleFeatureVectors, std::vector<int>& targetOutputs);
    void trainEnsembleModel(std::vector< FeatureVector > samples, std::vector<int> targetOutputs, std::vector<double> classWeights);
    static std::vector<double> ESVM::calcClassWeightsFromMode(int positivesCount, int negativesCount);
    static svm_node* getFeatureVector(FeatureVector features);
    static svm_node* getFeatureVector(double* features, int featureCount);
    svm_model* ensembleModel = nullptr;
    std::string targetID;
};

#endif/*ESVM_LIBSVM_H*/
