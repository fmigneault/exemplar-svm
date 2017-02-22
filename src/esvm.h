#ifndef ESVM_LIBSVM_H
#define ESVM_LIBSVM_H

#include "esvmTypes.h"
#include "svm.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class ESVM
{
public:
    inline ESVM() {}
    ESVM(std::vector<FeatureVector> positives, std::vector<FeatureVector> negatives, std::string id = "");
    ESVM(std::string trainingSamplesFilePath, std::string id = "");
    ESVM(svm_model* trainedModel, std::string id = "");   
    void readSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs);
    void readSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors);
    bool saveModelFile(std::string modelFilePath);
    double predict(FeatureVector probeSample);
    std::vector<double> predict(std::vector<FeatureVector> probeSamples);
    std::vector<double> predict(std::string probeSamplesFilePath, std::vector<int>* probeGroundTruths = nullptr);
    inline std::string getTargetID() { return targetID; }

private:
    void trainEnsembleModel(std::vector<FeatureVector> samples, std::vector<int> targetOutputs, std::vector<double> classWeights);
    static std::vector<double> calcClassWeightsFromMode(int positivesCount, int negativesCount);
    static svm_node* getFeatureVector(FeatureVector features);
    static svm_node* getFeatureVector(double* features, int featureCount);
    svm_model* ensembleModel = nullptr;
    std::string targetID;
};

#endif/*ESVM_LIBSVM_H*/
