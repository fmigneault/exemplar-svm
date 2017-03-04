#ifndef ESVM_LIBSVM_H
#define ESVM_LIBSVM_H

#include "esvmTypes.h"
#include "svm.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

enum FileFormat { BINARY, LIBSVM };

class ESVM
{
public:
    inline ESVM() {}
    ESVM(std::vector<FeatureVector> positives, std::vector<FeatureVector> negatives, std::string id = "");
    ESVM(std::vector<FeatureVector> samples, std::vector<int> targetOutputs, std::string id = "");
    ESVM(std::string trainingSamplesFilePath, std::string id = "");
    ESVM(svm_model* trainedModel, std::string id = "");   
    ~ESVM();
    bool isModelTrained();
    void readSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, 
                            std::vector<int>& targetOutputs, FileFormat format = LIBSVM);
    void readSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, FileFormat format = LIBSVM);
    void writeSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, 
                             std::vector<int>& targetOutputs, FileFormat format = LIBSVM);
    bool loadModelFile(std::string modelFilePath, FileFormat format = LIBSVM, std::string id = "");
    bool saveModelFile(std::string modelFilePath, FileFormat format = LIBSVM);
    double predict(FeatureVector probeSample);
    std::vector<double> predict(std::vector<FeatureVector> probeSamples);
    std::vector<double> predict(std::string probeSamplesFilePath, std::vector<int>* probeGroundTruths = nullptr);
    std::string targetID;

private:
    void trainModel(std::vector<FeatureVector> samples, std::vector<int> targetOutputs, std::vector<double> classWeights);
    static std::vector<double> calcClassWeightsFromMode(int positivesCount, int negativesCount);
    void loadModelFile_binary(std::string filePath);
    void saveModelFile_binary(std::string filePath);
    static void readSampleDataFile_binary(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs);
    static void readSampleDataFile_libsvm(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs);
    static void writeSampleDataFile_binary(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs);
    static void writeSampleDataFile_libsvm(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs);
    static FeatureVector getFeatureVector(svm_node* features);
    static svm_node* getFeatureNodes(FeatureVector features);
    static svm_node* getFeatureNodes(double* features, int featureCount);
    svm_model* model = nullptr;    
};

#endif/*ESVM_LIBSVM_H*/
