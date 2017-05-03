#ifndef ESVM_LIBSVM_H
#define ESVM_LIBSVM_H

#include "esvmTypes.h"
#include "svm.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <memory>

/* ESVM */

class ESVM
{
public:
    ~ESVM();
    ESVM();
    ESVM(const ESVM& esvm);
    /*ESVM(ESVM&& esvm);*/
    ESVM& operator=(const ESVM& esvm);
    ESVM(std::vector<FeatureVector> positives, std::vector<FeatureVector> negatives, std::string id = "");
    ESVM(std::vector<FeatureVector> samples, std::vector<int> targetOutputs, std::string id = "");
    ESVM(std::string trainingSamplesFilePath, std::string id = "");
    ESVM(svm_model* trainedModel, std::string id = "");
    static svm_model* makeEmptyModel();
    static void destroyModel(svm_model** model);
    bool isModelSet() const;
    bool isModelTrained() const;
    void logModelParameters(bool displaySV = false) const;
    static bool checkModelParameters(svm_model* model);
    static void readSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, 
                                   std::vector<int>& targetOutputs, FileFormat format = LIBSVM);
    static void readSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, FileFormat format = LIBSVM);
    static void writeSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors,
                                    std::vector<int>& targetOutputs, FileFormat format = LIBSVM);
    bool loadModelFile(std::string modelFilePath, FileFormat format = LIBSVM, std::string id = "");    
    bool saveModelFile(std::string modelFilePath, FileFormat format = LIBSVM) const;
    double predict(FeatureVector probeSample) const;
    std::vector<double> predict(std::vector<FeatureVector> probeSamples) const;
    std::vector<double> predict(std::string probeSamplesFilePath, std::vector<int>* probeGroundTruths = nullptr) const;
    std::string targetID;    

private:
    static void logModelParameters(svm_model* model, std::string id = "", bool displaySV = false);
    static bool checkBinaryHeader(std::ifstream& binaryFileStream, std::string header);
    static void checkModelParameters_assert(svm_model* model);
    static std::vector<double> calcClassWeightsFromMode(int positivesCount, int negativesCount);
    void trainModel(std::vector<FeatureVector> samples, std::vector<int> targetOutputs, std::vector<double> classWeights);
    void loadModelFile_libsvm(std::string filePath);
    void loadModelFile_binary(std::string filePath);
    void saveModelFile_binary(std::string filePath) const;
    static void readSampleDataFile_binary(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs);
    static void readSampleDataFile_libsvm(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs);
    static void writeSampleDataFile_binary(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs);
    static void writeSampleDataFile_libsvm(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs);
    static FeatureVector getFeatureVector(svm_node* features);
    static svm_node* getFeatureNodes(FeatureVector features);
    static svm_node* getFeatureNodes(double* features, int featureCount);
    static svm_model* deepCopyModel(svm_model* model = nullptr);
    static void removeTrainedModelUnusedData(svm_model* model, svm_problem* problem);
    void resetModel(svm_model* model = nullptr, bool copy = true);
    svm_model *esvmModel = nullptr;
    /*std::unique_ptr<svm_model> esvmModel = nullptr;*/
};

#endif/*ESVM_LIBSVM_H*/
