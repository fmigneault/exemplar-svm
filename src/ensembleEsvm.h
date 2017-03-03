#ifndef ENSEMBLEESVM_LIBSVM_H
#define ENSEMBLEESVM_LIBSVM_H

#include "esvm.h"
#include "esvmTypes.h"
#include "svm.h"
#include "mvector.hpp"      // Multi-Dimension vectors
#include "feHOG.h"
#include "esvmOptions.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class EnsembleESVM
{
public:
    EnsembleESVM();
    // EnsembleESVM(std::vector<FeatureVector> positives, std::vector<FeatureVector> negatives, std::string id = "");
    // EnsembleESVM(std::string trainingSamplesFilePath, std::string id = "");
    // EnsembleESVM(svm_model* trainedModel, std::string id = "");   
    // void readSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, 
    //                         std::vector<int>& targetOutputs, FileFormat format = LIBSVM);
    // void readSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, FileFormat format = LIBSVM);
    // void writeSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, 
    //                          std::vector<int>& targetOutputs, FileFormat format = LIBSVM);
    // bool saveModelFile(std::string modelFilePath);
    std::vector<double> predict(const cv::Mat roi);
    inline std::string getTargetID() { return targetID; }

private:
    // void trainEnsembleModel(std::vector<FeatureVector> samples, std::vector<int> targetOutputs, std::vector<double> classWeights);
    // static std::vector<double> calcClassWeightsFromMode(int positivesCount, int negativesCount);
    // static void readSampleDataFile_binary(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs);
    // static void readSampleDataFile_libsvm(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs);
    // static void writeSampleDataFile_binary(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs);
    // static void writeSampleDataFile_libsvm(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, std::vector<int>& targetOutputs);
    // static svm_node* getFeatureVector(FeatureVector features);
    // static svm_node* getFeatureVector(double* features, int featureCount);
    // svm_model* ensembleModel = nullptr;
    std::string targetID;
    std::vector<ESVM> ensembleEsvm;
    std::vector<FeatureVector> probeSampleFeats;
    size_t nPatches;
    cv::Size imageSize;
    cv::Size patchCounts;
    cv::Size blockSize;
    cv::Size blockStride;
    cv::Size cellSize;
    int nBins;

     xstd::mvector<1, double> scores;
     xstd::mvector<1, double> classificationScores;

     FeatureExtractorHOG hog;

    double hogHardcodedFoundMin;            // Min found using 'FullChokePoint' test with SAMAN pre-generated files
    double hogHardcodedFoundMax;     // Max found using 'FullChokePoint' test with SAMAN pre-generated files

};

#endif/*ENSEMBLEESVM_LIBSVM_H*/
