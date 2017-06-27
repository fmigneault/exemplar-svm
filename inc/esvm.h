#ifndef ESVM_LIBSVM_H
#define ESVM_LIBSVM_H

#include "svm.h"
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
#include <vector>
using namespace std;

/* ESVM */

class ESVM
{
public:    
    ~ESVM();
    ESVM();
    ESVM(const ESVM& esvm);
    /*ESVM(ESVM&& esvm);*/        
    ESVM(vector<FeatureVector> positives, vector<FeatureVector> negatives, string id = "");
    ESVM(vector<FeatureVector> samples, vector<int> targetOutputs, string id = "");
    ESVM(string trainingSamplesFilePath, string id = "");
    ESVM(svm_model* trainedModel, string id = "");
    static svm_model* makeEmptyModel();
    static void destroyModel(svm_model** model);
    bool isModelSet() const;
    bool isModelTrained() const;
    void logModelParameters(bool displaySV = false) const;
    static bool checkModelParameters(svm_model* model);
    static void readSampleDataFile(string filePath, vector<FeatureVector>& sampleFeatureVectors, 
                                   vector<int>& targetOutputs, FileFormat format = LIBSVM);
    static void readSampleDataFile(string filePath, vector<FeatureVector>& sampleFeatureVectors, FileFormat format = LIBSVM);
    static void writeSampleDataFile(string filePath, vector<FeatureVector>& sampleFeatureVectors,
                                    vector<int>& targetOutputs, FileFormat format = LIBSVM);
    bool loadModelFile(string modelFilePath, FileFormat format = LIBSVM, string id = "");    
    bool saveModelFile(string modelFilePath, FileFormat format = LIBSVM) const;
    double predict(FeatureVector probeSample) const;
    vector<double> predict(vector<FeatureVector> probeSamples) const;
    vector<double> predict(string probeSamplesFilePath, vector<int>* probeGroundTruths = nullptr) const;    
    string targetID;

    ESVM& operator=(ESVM esvm); // copy ctor
    ESVM(ESVM&& esvm);          // move ctor
    void swap(ESVM& esvm1, ESVM& esvm2)
    {
        using std::swap;
        swap(esvm1.esvmModel, esvm2.esvmModel);
    }

private:
    static void logModelParameters(svm_model* model, string id = "", bool displaySV = false);    
    static void checkModelParameters_assert(svm_model* model);
    static vector<double> calcClassWeightsFromMode(int positivesCount, int negativesCount);
    void trainModel(vector<FeatureVector> samples, vector<int> targetOutputs, vector<double> classWeights);
    void loadModelFile_libsvm(string filePath);
    void loadModelFile_binary(string filePath);
    void saveModelFile_binary(string filePath) const;
    static FeatureVector getFeatureVector(svm_node* features);
    static svm_node* getFeatureNodes(FeatureVector features);
    static svm_node* getFeatureNodes(double* features, int featureCount);
    static svm_model* deepCopyModel(svm_model* model = nullptr);
    static void removeTrainedModelUnusedData(svm_model* model, svm_problem* problem);
    void resetModel(svm_model* model = nullptr, bool copy = true);
    svm_model *esvmModel = nullptr;
    /*unique_ptr<svm_model> esvmModel = nullptr;*/    
};

#endif/*ESVM_LIBSVM_H*/
