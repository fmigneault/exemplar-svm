#ifndef ESVM_ENSEMBLE_H
#define ESVM_ENSEMBLE_H

#include "esvm.h"
#include "esvmTypes.h"
#include "mvector.hpp"
#include "feHOG.h"
#include "esvmOptions.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class esvmEnsemble
{
public:
    esvmEnsemble() {};
    esvmEnsemble(const std::vector<std::vector<cv::Mat> >& positiveROIs, const std::string negativesDir,
                 const std::vector<std::string>& positiveIDs = {}, const std::vector<std::vector<cv::Mat> >& additionalNegativeROIs = {});
    std::vector<double> predict(const cv::Mat& roi);
    bool saveModels(const std::string& saveDirectory);
    inline size_t getPositiveCount() { return enrolledPositiveIDs.size(); }
    inline size_t getPatchCount() { return patchCounts.area(); }
    inline std::string getPositiveID(int positiveIndex);

private:
    void setConstants(std::string negativesDir);
    std::vector<std::string> enrolledPositiveIDs;

    // Constants
    cv::Size imageSize;
    cv::Size patchCounts;
    cv::Size windowSize;
    cv::Size blockSize;
    cv::Size blockStride;
    cv::Size cellSize;
    int nBins;
    FeatureExtractorHOG hog;

    xstd::mvector<2, ESVM> EoESVM;

    std::string sampleFileExt;
    FileFormat sampleFileFormat;

    /* --- Reference 'hardcoded' feature normalization values --- */

    #if   ESVM_FEATURE_NORM_MODE == 1  // Min-Max features - overall normalization - across patches
    double hogMin;
    double hogMax;
    #elif ESVM_FEATURE_NORM_MODE == 2  // Z-Score features - overall normalization - across patches
    double hogMean;
    double hogStdDev;
    #elif ESVM_FEATURE_NORM_MODE == 3  // Min-Max features - per feature normalization - across patches
    FeatureVector hogMin;
    FeatureVector hogMax;
    #elif ESVM_FEATURE_NORM_MODE == 4  // Z-Score features - per feature normalization - across patches
    FeatureVector hogMean;
    FeatureVector hogStdDev;
    #elif ESVM_FEATURE_NORM_MODE == 5  // Min-Max features - overall normalization - for each patch
    std::vector<double> hogMin;
    std::vector<double> hogMax;
    #elif ESVM_FEATURE_NORM_MODE == 6  // Z-Score features - overall normalization - for each patch
    std::vector<double> hogMean;
    std::vector<double> hogStdDev;
    #elif ESVM_FEATURE_NORM_MODE == 7  // Min-Max features - per feature normalization - for each patch
    std::vector<FeatureVector> hogMin;
    std::vector<FeatureVector> hogMax;
    #elif ESVM_FEATURE_NORM_MODE == 8  // Z-Score features - per feature normalization - for each patch
    std::vector<FeatureVector> hogMean;
    std::vector<FeatureVector> hogStdDev;
    #endif/*ESVM_FEATURE_NORM_MODE*/

    /* --- Feature indexes to generate ramdom subspaces --- */

    #if ESVM_RANDOM_SUBSPACE_METHOD > 0    
    xstd::mvector<2, int> rsmFeatureIndexes;
    #endif/*ESVM_RANDOM_SUBSPACE_METHOD*/

    /* --- Reference 'hardcoded' score normalization values --- */

    #if   ESVM_SCORE_NORM_MODE == 1    // Min-Max scores normalization only post-fusion
    double scoreMinFusion;
    double scoreMaxFusion;
    #elif ESVM_SCORE_NORM_MODE == 2    // Z-Score scores normalization only post-fusion
    double scoreMean;
    double scoreStdDev;
    #elif ESVM_SCORE_NORM_MODE == 3    // Min-Max scores normalization only pre-fusion
    std::vector<double> scoreMinSVM;
    std::vector<double> scoreMaxSVM;
    #elif ESVM_SCORE_NORM_MODE == 4    // Z-Score scores normalization only pre-fusion
    std::vector<double> scoreMeanSVM;
    std::vector<double> scoreStdDevSVM;
    #elif ESVM_SCORE_NORM_MODE == 5    // Min-Max scores normalization both pre/post-fusion
    std::vector<double> scoreMinSVM;
    std::vector<double> scoreMaxSVM;
    double scoreMinFusion;
    double scoreMaxFusion;
    #elif ESVM_SCORE_NORM_MODE == 6    // Z-Score scores normalization both pre/post-fusion
    std::vector<double> scoreMeanSVM;
    std::vector<double> scoreStdDevSVM;
    double scoreMeanFusion;
    double scoreStdDevFusion;
    #endif/*ESVM_SCORE_NORM_MODE*/
};

#endif/*ESVM_ENSEMBLE_H*/
