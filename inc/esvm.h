#ifndef ESVM_LIBSVM_H
#define ESVM_LIBSVM_H

#include "esvmTypes.h"

#include "datafile.h"
#include "generic.h"
#include "types.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace esvm {

class ESVM
{
public:
    // contructors/destructors
    ~ESVM();
    ESVM();
    ESVM(const ESVM& esvm);
    ESVM(std::vector<FeatureVector> positives, std::vector<FeatureVector> negatives, std::string id = "");
    ESVM(std::vector<FeatureVector> samples, std::vector<int> targetOutputs, std::string id = "");
    ESVM(std::string trainingSamplesFilePath, std::string id = "");
    ESVM(svmModel* trainedModel, std::string id = "");
    ESVM& operator=(ESVM esvm); // copy ctor
    ESVM(ESVM&& esvm);          // move ctor
    void swap(ESVM& esvm1, ESVM& esvm2)
    {
        std::swap(esvm1.esvmModel, esvm2.esvmModel);
        std::swap(esvm1.ID, esvm2.ID);
    }
    // instance methods
    bool isModelSet() const;
    bool isModelTrained() const;
    void logModelParameters(bool displaySV = false) const;
    bool loadModelFile(std::string modelFilePath, FileFormat format = LIBSVM, std::string id = "");
    bool saveModelFile(std::string modelFilePath, FileFormat format = LIBSVM) const;
    double predict(FeatureVector probeSample) const;
    std::vector<double> predict(std::vector<FeatureVector> probeSamples) const;
    std::vector<double> predict(std::string probeSamplesFilePath, std::vector<int>* probeGroundTruths = nullptr) const;
    // static methods
    static svmModel* makeEmptyModel();
    static void destroyModel(svmModel** model);
    static bool checkModelParameters(svmModel* model);
    static void readSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors,
                                   std::vector<int>& targetOutputs, FileFormat format = LIBSVM);
    static void readSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors, FileFormat format = LIBSVM);
    static void writeSampleDataFile(std::string filePath, std::vector<FeatureVector>& sampleFeatureVectors,
                                    std::vector<int>& targetOutputs, FileFormat format = LIBSVM);
    // properties
    std::string ID;

private:
    // instance methods
    void trainModel(std::vector<FeatureVector> samples, std::vector<int> targetOutputs, std::vector<double> classWeights);
    void loadModelFile_libsvm(std::string filePath);
    void loadModelFile_binary(std::string filePath);
    void saveModelFile_binary(std::string filePath) const;
    void resetModel(svmModel* model = nullptr, bool copy = true);
    // static methods
    static void logModelParameters(svmModel* model, std::string id = "", bool displaySV = false);
    static void checkModelParameters_assert(svmModel* model);
    static std::vector<double> calcClassWeightsFromMode(int positivesCount, int negativesCount);
    static FeatureVector getFeatureVector(svmFeature* features);
    static svmFeature* getFeatureNodes(FeatureVector features);
    static svmFeature* getFeatureNodes(double* features, int featureCount);
    static svmModel* deepCopyModel(svmModel* model = nullptr);
    static void removeTrainedModelUnusedData(svmModel* model, svmProblem* problem);
    static FreeModelState getFreeSV(svmModel* model);
    // object
    svmModel *esvmModel = nullptr;
    /*unique_ptr<svmModel> esvmModel = nullptr;*/
};

} // namespace esvm

#endif/*ESVM_LIBSVM_H*/
